from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
import torch.nn as nn


@dataclass
class NodeFieldSpec:
    name: str
    cardinality: int
    emb_dim: int


class BondBackbone(nn.Module):
    """
    Minimal GraphSAGE-style encoder:
      - Node embedding = [id_emb | categorical embs | numeric MLP]
      - Aggregate by issuer + sector mean pools over *batch-present* peers
    """
    def __init__(
        self,
        n_nodes: int,
        node_id_dim: int,
        cat_specs: List[NodeFieldSpec],
        n_num: int,
        num_hidden: int = 64,
        out_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.id_emb = nn.Embedding(n_nodes, node_id_dim)
        self.cat_embs = nn.ModuleDict(
            {s.name: nn.Embedding(s.cardinality, s.emb_dim) for s in cat_specs}
        )
        self.n_num = n_num
        if n_num > 0:
            self.num_mlp = nn.Sequential(
                nn.Linear(n_num, num_hidden),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(num_hidden, num_hidden),
                nn.ReLU(),
            )
            num_out = num_hidden
        else:
            self.num_mlp = None
            num_out = 0

        in_dim = node_id_dim + sum(s.emb_dim for s in cat_specs) + num_out
        self.agg = nn.Sequential(
            nn.Linear(in_dim * 3, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(out_dim, out_dim),
        )
        self.out_dim = out_dim

    def _self_repr(
        self,
        node_ids: torch.Tensor,
        cats: Dict[str, torch.Tensor],
        nums: Optional[torch.Tensor],
    ) -> torch.Tensor:
        chunks = [self.id_emb(node_ids)]
        for name, emb in self.cat_embs.items():
            chunks.append(emb(cats[name]))
        if self.num_mlp is not None and nums is not None:
            chunks.append(self.num_mlp(nums))
        return torch.cat(chunks, dim=1)

    @staticmethod
    def _mean_pool_batch_peers(
        h_self: torch.Tensor,
        node_ids: torch.Tensor,
        groups: Dict[int, torch.Tensor],
        keys: torch.Tensor,
    ) -> torch.Tensor:
        """
        For each sample i, mean-pool h_self over batch nodes that share the same group id 'keys[i]'.
        Ensures all tensors are on the same device.
        """
        device = node_ids.device
        outs = []
        for k in keys.tolist():
            gnodes = groups[int(k)].to(device)
            mask = torch.isin(node_ids, gnodes)  # [B]
            # mask should include at least the current node; safe-guard anyway
            if not mask.any():
                outs.append(h_self.mean(0))  # fallback
            else:
                outs.append(h_self[mask].mean(0))
        return torch.stack(outs, dim=0)

    def forward(
        self,
        node_ids: torch.Tensor,            # [B]
        cats: Dict[str, torch.Tensor],     # each [B]
        nums: Optional[torch.Tensor],      # [B, n_num] or None
        issuer_groups: Dict[int, torch.Tensor],
        sector_groups: Dict[int, torch.Tensor],
        node_to_issuer: torch.Tensor,      # [N]
        node_to_sector: torch.Tensor,      # [N]
    ) -> torch.Tensor:
        # representations for batch nodes
        h_self = self._self_repr(node_ids, cats, nums)          # [B, D0]

        # map batch node ids to issuer/sector ids
        issuers = node_to_issuer[node_ids]                      # [B]
        sectors = node_to_sector[node_ids]                      # [B]

        # mean-pool over peers (present within this batch) that share issuer/sector
        h_iss = self._mean_pool_batch_peers(h_self, node_ids, issuer_groups, issuers)  # [B, D0]
        h_sec = self._mean_pool_batch_peers(h_self, node_ids, sector_groups, sectors)  # [B, D0]

        return self.agg(torch.cat([h_self, h_iss, h_sec], dim=1))  # [B, out_dim]


from .portfolio_encoder import PMAPooling, TargetPortfolioCrossAttention

class LiquidityResidualBackbone(nn.Module):
    """
    Target/basket reasoning:
      - PMA summary of portfolio tokens (DV01-weighted values)
      - Target→portfolio cross-attention
      - Fuse [target, context, fused] and predict via heads
    """
    def __init__(self, d_model: int = 128, n_heads: int = 4, dropout: float = 0.1, heads_module=None):
        super().__init__()
        self.pma   = PMAPooling(d_model, n_heads, dropout)
        self.cross = TargetPortfolioCrossAttention(d_model, n_heads, dropout)
        self.fuse_ln = nn.LayerNorm(3*d_model)
        self.fuse    = nn.Sequential(nn.Linear(3*d_model, d_model),
                                     nn.ReLU(), nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                                     nn.Linear(d_model, d_model))
        if heads_module is None:
            from .heads import QuantileHeads
            heads_module = QuantileHeads
        self.heads = heads_module(d_model, hidden=256, dropout=dropout)

    def forward_from_node_embeddings(
        self,
        node_embeddings: torch.Tensor,           # (N, d)
        target_index: torch.LongTensor,          # (B,)
        port_index: torch.LongTensor,            # (T,)
        port_batch: torch.LongTensor,            # (T,) in [0..B-1]
        port_weight: torch.Tensor | None = None  # (T,)
    ) -> Dict[str, torch.Tensor]:
        B = target_index.numel()
        d = node_embeddings.size(-1)
        device = node_embeddings.device

        targets  = node_embeddings[target_index]                  # (B,d)
        contexts = torch.zeros(B, d, device=device)
        fused    = torch.zeros(B, d, device=device)

        for b in range(B):
            mask = (port_batch == b)
            if mask.any():
                tokens = node_embeddings[port_index[mask]]        # (n_i, d)
                w      = port_weight[mask] if port_weight is not None else None
                contexts[b] = self.pma(tokens, w)
                fused[b]    = self.cross(targets[b], tokens, w)
            else:
                contexts[b] = torch.zeros(d, device=device)
                fused[b]    = targets[b]

        z = torch.cat([targets, contexts, fused], dim=-1)
        z = self.fuse_ln(z)
        z = self.fuse(z)
        return self.heads(z)
