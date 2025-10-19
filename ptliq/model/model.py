from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import NodeFieldSpec
from .portfolio_encoder import PortfolioEncoder
from .utils import resolve_device
from .heads import QuantileHead

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
    # Encoder inits that passed tests:
    enc_beta_init: float = 0.8
    enc_learnable_tau: bool = False
    enc_tau_init: float = 1.0

    # Fusion / gate
    use_fused_aggregates: bool = True
    ctx_gate_init: float = 0.0  # tests keep 0.0; prod can start >0

    # Heads / loss
    use_quantiles: bool = False  # tests keep False
    pinball_taus: List[float] = None  # e.g. [0.5, 0.9] in prod
    loss_type: str = "mse"  # "mse" (tests) or "huber_pinball"
    huber_delta: float = 0.1  # for smooth_l1
    monotone_reg: float = 0.0  # reserved for provider calibrators


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
        #return H.reshape(*node_ids.shape, -1)
        # compute the final embedding size explicitly
        if H.ndim == 2:
            out_dim = H.shape[-1]
        else:
            # fallback: id + cats dims (shouldn’t be needed if H is [*, C])
            out_dim = self.id_emb.embedding_dim
            for emb in self.cat_embs.values():
                out_dim += emb.embedding_dim

        # unflatten without using -1 so zero-sized shapes are OK
        return H.view(*node_ids.shape, out_dim)


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
            nhead=cfg.nhead,  # kept for API compat (unused in our impl)
            num_layers=max(1, cfg.n_layers),
            dropout=cfg.tr_dropout,
            beta_init=cfg.enc_beta_init,
            learnable_tau=cfg.enc_learnable_tau,
            tau_init=cfg.enc_tau_init,
        )

        # head over [trade, attn_ctx, masked_mean_ctx]
        self.head = RegressionHead(d_in=cfg.d_model * 3, hidden=cfg.head_hidden, dropout=cfg.head_dropout)

        # Optional quantiles (off for tests)
        self.quant_head = None
        if cfg.use_quantiles and cfg.pinball_taus:
            self.quant_head = QuantileHead(d_in=cfg.d_model * 3, hidden=cfg.head_hidden, taus=cfg.pinball_taus, dropout=cfg.head_dropout)

        # Fusion gate
        self.ctx_gate = nn.Parameter(torch.tensor(cfg.ctx_gate_init))
        self.post_norm = nn.LayerNorm(cfg.d_model * 3)

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
        # Ensure padded positions (-1) don't index tensors; clamp to 0 for safe gathers
        port_nodes_safe = torch.clamp(port_nodes, min=0)
        cats_p = self._build_portfolio_cats(port_nodes_safe, node_to_sector)
        H_pc = self.shared(port_nodes_safe, cats_p)   # (B, T, C)
        H_p = self.port_proj(H_pc)                    # (B, T, D)

        valid_mask = self._make_valid_mask(port_len, T)    # True=valid

        # cross-attention context
        ctx_attn = self.port_enc(h_t, H_p, port_len)       # (B, D)

        # feature-aware masked mean (sector match)
        sector_q = batch_cats.get("sector_code", None)
        if sector_q is not None:
            ctx_masked = self._masked_sector_mean(H_p, valid_mask, port_nodes_safe, node_to_sector, sector_q)
        else:
            # fallback to valid mean if sector absent
            w = valid_mask.float().unsqueeze(-1)
            ctx_masked = (H_p * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)


        # gated mix (keeps tests unaffected with gate=0.0)
        gate = torch.sigmoid(self.ctx_gate) if self.cfg.use_fused_aggregates else torch.tensor(0.0, device=h_t.device)
        ctx_mix = ctx_attn + gate * (ctx_masked - ctx_attn)

        # fused features (normalized) -> mean
        x = torch.cat([h_t, ctx_attn, ctx_mix], dim=-1)
        x = self.post_norm(x)
        mean = self.head(x)
        if self.calibrator is not None:
            mean = mean * self.calibrator

        out = {"mean": mean}

        # optional quantiles
        if self.quant_head is not None:
            q = self.quant_head(x)  # (B, len(taus))
            out["quantiles"] = q
            out["taus"] = torch.tensor(self.cfg.pinball_taus, device=q.device)
        return out

    def compute_loss(self, out: Dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        # Switchable loss: MSE for tests, Huber+Pinball for prod
        pred = out["mean"].squeeze(-1).float()
        y = target.to(pred.device).float().view_as(pred)

        if self.cfg.loss_type.lower() == "mse":
            return torch.mean((pred - y) ** 2)

        # Huber on mean
        huber = F.smooth_l1_loss(pred, y, beta=self.cfg.huber_delta)

        # Pinball on quantiles (if present)
        pinball = 0.0
        if ("quantiles" in out) and ("taus" in out):
            q = out["quantiles"]         # (B, K)
            taus = out["taus"].view(1, -1)  # (1, K)
            diff = y.unsqueeze(1) - q        # (B, K)
            pinball = torch.mean(torch.maximum(taus * diff, (taus - 1.0) * diff))

        return huber + pinball * 1.0 + self.cfg.monotone_reg * 0.0  # placeholder for calibrator reg


def H_tc_needs_squeeze(shared: IdCatEmbedding, node_ids: torch.Tensor) -> bool:
    # If user passes shape (B,) we get (B, C); if (B,1) we get (B,1,C).
    # Handle both robustly.
    return node_ids.ndim == 2 and node_ids.shape[1] == 1


# -----------------------------
# New relation-aware GATv2 encoder and full model wrapper (Colab backbone)
# -----------------------------
from typing import Optional, Dict
import torch
import torch.nn as nn
try:
    from torch_geometric.data import Data
    from torch_geometric.nn import GATv2Conv, BatchNorm as PygBatchNorm
except Exception:  # pragma: no cover - allow import in environments without PyG during non-GNN tests
    Data = object  # type: ignore
    GATv2Conv = object  # type: ignore
    class PygBatchNorm(nn.Module):  # minimal stub
        def __init__(self, dim: int):
            super().__init__()
        def forward(self, x):
            return x

from .backbone import LiquidityResidualBackbone

class GraphEncoder(nn.Module):
    """
    Encodes node features with relation-aware GATv2:
      - Learn an embedding for relation_id
      - Edge attributes = [rel_emb, scaled edge_weight] with learnable per-relation gain
      - Optional issuer embedding concatenated to x
    """
    def __init__(self,
                 x_dim: int,
                 num_relations: int,
                 d_model: int = 128,
                 issuer_emb_dim: int = 16,
                 num_issuers: Optional[int] = None,
                 heads: int = 4,
                 rel_emb_dim: int = 16,
                 dropout: float = 0.1,
                 rel_init_boost: dict | None = None):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.heads   = heads

        # relation embedding + log gain per relation
        self.rel_emb = nn.Embedding(num_relations, rel_emb_dim)
        self.rel_log_gain = nn.Parameter(torch.zeros(num_relations))
        if rel_init_boost:
            with torch.no_grad():
                for rid, gain in rel_init_boost.items():
                    rid_i = int(rid)
                    if 0 <= rid_i < num_relations:
                        self.rel_log_gain[rid_i] = float(torch.log(torch.tensor(gain)))

        # optional issuer embedding
        self.use_issuer = issuer_emb_dim > 0
        self.issuer_emb_dim = issuer_emb_dim
        self._issuer_fixed_card = num_issuers
        self.issuer_emb: Optional[nn.Embedding] = None
        if self.use_issuer and (num_issuers is not None):
            self.issuer_emb = nn.Embedding(num_issuers + 1, issuer_emb_dim)  # +1 for unknown

        in_dim = x_dim + (issuer_emb_dim if self.use_issuer else 0)
        self.in_proj = nn.Linear(in_dim, d_model)
        self.in_bn   = PygBatchNorm(d_model)

        edge_dim = rel_emb_dim + 1  # concat([rel_emb, edge_weight])
        # Ensure GAT output matches BatchNorm size exactly
        assert heads > 0, "heads must be a positive integer"
        assert d_model % heads == 0, f"d_model={d_model} must be divisible by heads={heads}"
        out_per_head = d_model // heads

        self.conv1 = GATv2Conv(d_model, out_per_head, heads=heads,
                               edge_dim=edge_dim, dropout=dropout, add_self_loops=False)
        self.bn1   = PygBatchNorm(d_model)
        self.conv2 = GATv2Conv(d_model, out_per_head, heads=heads,
                               edge_dim=edge_dim, dropout=dropout, add_self_loops=False)
        self.bn2   = PygBatchNorm(d_model)

    def _edge_attr(self, edge_type: torch.Tensor, edge_weight: Optional[torch.Tensor]) -> torch.Tensor:
        rel = self.rel_emb(edge_type.long())                  # (E, rel_emb_dim)
        if edge_weight is None:
            w = torch.zeros(edge_type.size(0), 1, device=rel.device, dtype=rel.dtype)
        else:
            w = edge_weight.view(-1, 1).to(rel.dtype)
        gain = torch.exp(self.rel_log_gain[edge_type.long()]).unsqueeze(-1)  # (E,1)
        w = w * gain
        return torch.cat([rel, w], dim=-1)                    # (E, rel_emb_dim+1)

    def _issuer_embed(self, issuer_index: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if not self.use_issuer:
            return None
        if self.issuer_emb is None:
            assert issuer_index is not None, "issuer_index required to init issuer embedding"
            num_issuers = int(issuer_index.max().item()) + 1 if issuer_index.numel() > 0 else 1
            self.issuer_emb = nn.Embedding(num_issuers + 1, self.issuer_emb_dim).to(issuer_index.device)
        idx = (issuer_index.long() + 1).clamp(min=0, max=self.issuer_emb.num_embeddings - 1)
        return self.issuer_emb(idx)

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_type: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None,
                issuer_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        iss = self._issuer_embed(issuer_index) if self.use_issuer else None
        x_in = torch.cat([x, iss], dim=-1) if iss is not None else x
        h = self.in_proj(x_in); h = self.in_bn(h); h = torch.relu(h); h = self.dropout(h)

        ea = self._edge_attr(edge_type, edge_weight)
        h = self.conv1(h, edge_index, edge_attr=ea); h = self.bn1(h); h = torch.relu(h); h = self.dropout(h)
        h = self.conv2(h, edge_index, edge_attr=ea); h = self.bn2(h); h = torch.relu(h)
        return h

class LiquidityModelGAT(nn.Module):
    """
    End-to-end portfolio-conditioned residual model:
      encoder(GraphEncoder) -> LiquidityResidualBackbone -> heads
    """
    def __init__(self,
                 x_dim: int,
                 num_relations: int,
                 d_model: int = 128,
                 issuer_emb_dim: int = 16,
                 num_issuers: Optional[int] = None,
                 dropout: float = 0.1,
                 heads: int = 4,
                 rel_emb_dim: int = 16,
                 rel_init_boost: dict | None = None):
        super().__init__()
        self.encoder  = GraphEncoder(
            x_dim, num_relations, d_model,
            issuer_emb_dim=issuer_emb_dim, num_issuers=num_issuers,
            heads=heads, rel_emb_dim=rel_emb_dim, dropout=dropout,
            rel_init_boost=rel_init_boost
        )
        self.backbone = LiquidityResidualBackbone(d_model=d_model, n_heads=heads, dropout=dropout)

    # convenience: tensors API
    def forward_from_tensors(self,
                             x: torch.Tensor, edge_index: torch.Tensor,
                             edge_type: torch.Tensor, edge_weight: Optional[torch.Tensor],
                             target_index: torch.LongTensor,
                             port_index: torch.LongTensor, port_batch: torch.LongTensor,
                             port_weight: Optional[torch.Tensor] = None,
                             issuer_index: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        h = self.encoder(x, edge_index, edge_type, edge_weight=edge_weight, issuer_index=issuer_index)
        return self.backbone.forward_from_node_embeddings(h, target_index, port_index, port_batch, port_weight)

    # convenience: PyG Data API
    def forward_from_data(self,
                          data: Data,
                          target_index: torch.LongTensor,
                          port_index: torch.LongTensor, port_batch: torch.LongTensor,
                          port_weight: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        h = self.encoder(
            data.x, data.edge_index, data.edge_type,
            edge_weight=getattr(data, "edge_weight", None),
            issuer_index=getattr(data, "issuer_index", None),
        )
        return self.backbone.forward_from_node_embeddings(h, target_index, port_index, port_batch, port_weight)

# Alias for notebooks
LiquidityModelColab = LiquidityModelGAT
