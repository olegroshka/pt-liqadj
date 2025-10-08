from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import NodeFieldSpec
from .portfolio_encoder import PortfolioEncoder
from .utils import resolve_device


@dataclass
class ModelConfig:
    n_nodes: int
    node_id_dim: int
    cat_specs: List[NodeFieldSpec]
    n_num: int

    gnn_num_hidden: int = 64
    gnn_out_dim: int = 128
    gnn_dropout: float = 0.0

    d_model: int = 128
    nhead: int = 4
    n_layers: int = 1
    tr_dropout: float = 0.0

    head_hidden: int = 128
    head_dropout: float = 0.0

    use_calibrator: bool = False
    use_baseline: bool = False
    device: str = "auto"


class IdCatEmbedding(nn.Module):
    """
    Shared node-id + categorical embeddings used by BOTH trade and portfolio tokens.
    Ensures the query/key live in the same space for effective dot-product attention.
    """
    def __init__(self, n_nodes: int, node_id_dim: int, cat_specs: List[NodeFieldSpec]):
        super().__init__()
        self.id_emb = nn.Embedding(n_nodes, node_id_dim)
        self.cat_embs = nn.ModuleDict()
        for spec in cat_specs:
            self.cat_embs[spec.name] = nn.Embedding(spec.cardinality, spec.emb_dim)
        self.out_dim = node_id_dim + sum(m.embedding_dim for m in self.cat_embs.values())

    def forward(self, node_ids: torch.Tensor, cats: Dict[str, torch.Tensor]) -> torch.Tensor:
        # node_ids: [...], cats[name]: [...]
        flat_ids = node_ids.reshape(-1)
        pieces = [self.id_emb(flat_ids)]
        for name, emb in self.cat_embs.items():
            flat_cat = cats[name].reshape(-1)
            pieces.append(emb(flat_cat))
        H = torch.cat(pieces, dim=-1)  # [*, out_dim]
        return H.reshape(*node_ids.shape, -1)


class NumericEncoder(nn.Module):
    def __init__(self, n_in: int, n_hidden: int):
        super().__init__()
        if n_in > 0:
            self.proj = nn.Sequential(
                nn.Linear(n_in, n_hidden),
                nn.GELU(),
            )
        else:
            self.proj = None

    def forward(self, x: Optional[torch.Tensor]) -> torch.Tensor:
        if self.proj is None:
            if x is None:
                return torch.zeros(0)
            B = x.shape[0] if x.ndim > 0 else 1
            return x.new_zeros((B, 0))
        if x is None:
            raise ValueError("NumericEncoder expects a tensor but received None.")
        if x.ndim == 3:
            x = x.reshape(-1, x.shape[-1])
        return self.proj(x)


class RegressionHead(nn.Module):
    def __init__(self, d_in: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PortfolioResidualModel(nn.Module):
    """
    Shared id+cat embeddings → trade/proj (+ numerics) and portfolio/proj (no numerics)
    → cross-attn → PLUS feature-aware masked mean over same-sector items
    → concat[trade, ctx_attn, ctx_masked] → regression.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        dev = resolve_device(cfg.device)

        # shared embeddings
        self.shared = IdCatEmbedding(cfg.n_nodes, cfg.node_id_dim, cfg.cat_specs)

        # trade branch: shared(id+cat) + numerics -> d_model
        self.num_enc = NumericEncoder(cfg.n_num, cfg.gnn_num_hidden) if cfg.n_num > 0 else None
        trade_in = self.shared.out_dim + (cfg.gnn_num_hidden if self.num_enc is not None else 0)
        self.trade_proj = nn.Sequential(
            nn.Linear(trade_in, cfg.d_model),
            nn.GELU(),
            nn.LayerNorm(cfg.d_model),
        )

        # portfolio items: shared(id+cat) only -> d_model
        self.port_proj = nn.Sequential(
            nn.Linear(self.shared.out_dim, cfg.d_model),
            nn.GELU(),
            nn.LayerNorm(cfg.d_model),
        )

        self.port_enc = PortfolioEncoder(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=max(1, cfg.n_layers),
            dropout=cfg.tr_dropout,
        )

        # head over [trade, attn_ctx, masked_mean_ctx]
        self.head = RegressionHead(d_in=cfg.d_model * 3, hidden=cfg.head_hidden, dropout=cfg.head_dropout)

        self.calibrator = None
        if cfg.use_calibrator:
            self.calibrator = nn.Parameter(torch.ones(()))

        self.to(dev)

    @staticmethod
    def _make_valid_mask(lengths: torch.Tensor, T: int) -> torch.Tensor:
        idx = torch.arange(T, device=lengths.device).unsqueeze(0).expand(lengths.shape[0], T)
        return idx < lengths.unsqueeze(1)  # True=valid

    def _build_portfolio_cats(
        self,
        port_nodes: torch.Tensor,
        node_to_sector: Optional[torch.Tensor],
        node_to_rating: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, T = port_nodes.shape
        dev = port_nodes.device
        cats_p: Dict[str, torch.Tensor] = {}
        for name in self.shared.cat_embs.keys():
            if name == "sector_code" and node_to_sector is not None:
                cats_p[name] = node_to_sector[port_nodes]  # (B, T)
            elif name == "rating_code" and node_to_rating is not None:
                cats_p[name] = node_to_rating[port_nodes]
            else:
                cats_p[name] = torch.zeros((B, T), dtype=torch.long, device=dev)
        return cats_p

    def _masked_sector_mean(
        self,
        H_p: torch.Tensor,                 # (B, T, D)
        valid_mask: torch.Tensor,          # (B, T) True=valid
        port_nodes: torch.Tensor,          # (B, T)
        node_to_sector: Optional[torch.Tensor],
        sector_q: torch.Tensor,            # (B,)
    ) -> torch.Tensor:
        if node_to_sector is None:
            # fall back to valid mean if mapping is missing
            w = valid_mask.float().unsqueeze(-1)  # (B,T,1)
            denom = w.sum(dim=1).clamp_min(1.0)
            return (H_p * w).sum(dim=1) / denom

        port_sector = node_to_sector[port_nodes]            # (B, T)
        same = (port_sector == sector_q.unsqueeze(1)) & valid_mask
        w = same.float().unsqueeze(-1)                      # (B, T, 1)
        denom = w.sum(dim=1).clamp_min(1.0)
        ctx = (H_p * w).sum(dim=1) / denom                  # (B, D)
        # If a row had zero matches, the denom clamp makes this an average of zeros.
        # Blend with valid mean to avoid degenerate zeros:
        any_match = (denom.squeeze(-1) > 0.5).float().unsqueeze(-1)  # (B,1)
        valid_w = valid_mask.float().unsqueeze(-1)
        valid_mean = (H_p * valid_w).sum(dim=1) / valid_w.sum(dim=1).clamp_min(1.0)
        return any_match * ctx + (1.0 - any_match) * valid_mean

    def forward(
        self,
        batch_nodes: torch.Tensor,                      # (B,)
        batch_cats: Dict[str, torch.Tensor],            # each (B,)
        batch_nums: Optional[torch.Tensor],             # (B, n_num)
        issuer_groups: Optional[Dict[int, torch.Tensor]],   # unused
        sector_groups: Optional[Dict[int, torch.Tensor]],   # unused
        node_to_issuer: Optional[torch.Tensor],             # unused
        node_to_sector: Optional[torch.Tensor],             # (N,)
        port_nodes: torch.Tensor,                       # (B, T)
        port_len: torch.Tensor,                         # (B,)
        size_side_urg: Optional[torch.Tensor],          # unused
        baseline_feats: Optional[torch.Tensor],         # unused
    ) -> Dict[str, torch.Tensor]:
        B, T = port_nodes.shape

        # trade representation
        H_tc = self.shared(batch_nodes, batch_cats).squeeze(1) if H_tc_needs_squeeze(self.shared, batch_nodes) else self.shared(batch_nodes, batch_cats)
        if H_tc.ndim == 3:  # (B,1,C) -> (B,C)
            H_tc = H_tc.squeeze(1)
        if self.num_enc is not None:
            H_t = torch.cat([H_tc, self.num_enc(batch_nums)], dim=-1)
        else:
            H_t = H_tc
        h_t = self.trade_proj(H_t)  # (B, D)

        # portfolio items
        cats_p = self._build_portfolio_cats(port_nodes, node_to_sector)
        H_pc = self.shared(port_nodes, cats_p)   # (B, T, C)
        H_p = self.port_proj(H_pc)               # (B, T, D)

        valid_mask = self._make_valid_mask(port_len, T)    # True=valid

        # cross-attention context
        ctx_attn = self.port_enc(h_t, H_p, port_len)       # (B, D)

        # feature-aware masked mean (sector match)
        sector_q = batch_cats.get("sector_code", None)
        if sector_q is not None:
            ctx_masked = self._masked_sector_mean(H_p, valid_mask, port_nodes, node_to_sector, sector_q)
        else:
            # fallback to valid mean if sector absent
            w = valid_mask.float().unsqueeze(-1)
            ctx_masked = (H_p * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)

        # predict
        x = torch.cat([h_t, ctx_attn, ctx_masked], dim=-1)  # (B, 3D)
        mean = self.head(x)
        if self.calibrator is not None:
            mean = mean * self.calibrator
        return {"mean": mean}

    def compute_loss(self, out: Dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        pred = out["mean"].squeeze(-1).float()
        y = target.to(pred.device).float().view_as(pred)
        # Huber tends to be friendlier on tiny synthetic sets
        return F.smooth_l1_loss(pred, y, beta=0.1)


def H_tc_needs_squeeze(shared: IdCatEmbedding, node_ids: torch.Tensor) -> bool:
    # If user passes shape (B,) we get (B, C); if (B,1) we get (B,1,C).
    # Handle both robustly.
    return node_ids.ndim == 2 and node_ids.shape[1] == 1
