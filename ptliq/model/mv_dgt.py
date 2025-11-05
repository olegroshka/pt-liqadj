# mv_dgt.py
from __future__ import annotations
from typing import Dict, Optional
import torch
from torch import nn, Tensor
from ptliq.utils.attn_utils import extract_heads_mean_std, attn_store_update
try:
    from torch_geometric.nn import TransformerConv
except Exception as e:  # pragma: no cover - optional dependency guard for import-time
    TransformerConv = None  # type: ignore

# --- Portfolio context → per-node vectors ------------------------------------
# Fully differentiable, leave-one-out, no dead vars.
@torch.no_grad()
def _check_port_ctx(port_ctx: dict):
    # abs weights are optional (fallback to abs(signed)) for backward-compat with older tests
    req = {"port_nodes_flat","port_w_signed_flat","port_len"}
    missing = req.difference(port_ctx.keys())
    if missing:
        raise ValueError(f"[portfolio] missing fields: {sorted(missing)}")


def _portfolio_vectors(H: Tensor, port_ctx: dict, l2_normalize: bool = True) -> tuple[Tensor, Tensor]:
    """
    H: (N, D) node embeddings. Returns two (N, D) tensors:
      - V_abs: average basket intensity vector per node (abs weights)
      - V_sgn: average basket direction vector per node (signed weights)
    Expects port_ctx with keys: port_nodes_flat, port_w_abs_flat, port_w_signed_flat, port_len.
    """
    _check_port_ctx(port_ctx)
    dev, dt = H.device, H.dtype

    nodes_flat: Tensor = port_ctx["port_nodes_flat"].to(dev)            # (L,)
    w_sgn: Tensor      = port_ctx["port_w_signed_flat"].to(dev).to(dt)   # (L,)
    # abs weights may be missing in older artifacts; fall back to abs(signed)
    if "port_w_abs_flat" in port_ctx:
        w_abs: Tensor = port_ctx["port_w_abs_flat"].to(dev).to(dt)
    else:
        w_abs = torch.abs(w_sgn)
    lens: Tensor       = port_ctx["port_len"].to(dev).long()             # (G,)

    gid: Tensor = torch.repeat_interleave(torch.arange(lens.numel(), device=dev, dtype=torch.long), lens)  # (L,)
    if not (nodes_flat.numel() == w_abs.numel() == w_sgn.numel() == gid.numel()):
        raise ValueError("[portfolio] inconsistent flattened lengths")

    N, D = H.size(0), H.size(1)
    G = lens.numel()

    # Gather node embeddings for each line item
    H_lines = H.index_select(0, nodes_flat)  # (L, D)

    # Basket-level sums
    P_abs = torch.zeros(G, D, device=dev, dtype=dt)
    P_abs.index_add_(0, gid, w_abs.unsqueeze(1) * H_lines)

    P_sgn = torch.zeros(G, D, device=dev, dtype=dt)
    P_sgn.index_add_(0, gid, w_sgn.unsqueeze(1) * H_lines)

    # Leave-one-out basket vectors per line
    P_abs_g = P_abs.index_select(0, gid)  # (L, D)
    P_sgn_g = P_sgn.index_select(0, gid)  # (L, D)

    loo_abs = P_abs_g - w_abs.unsqueeze(1) * H_lines
    loo_sgn = P_sgn_g - w_sgn.unsqueeze(1) * H_lines

    # Contribution to the node from each basket membership, weighted by own |w_abs|
    contrib_abs = w_abs.unsqueeze(1) * loo_abs  # (L, D)
    contrib_sgn = w_abs.unsqueeze(1) * loo_sgn  # (L, D)

    # Aggregate to nodes
    V_abs = torch.zeros(N, D, device=dev, dtype=dt)
    V_abs.index_add_(0, nodes_flat, contrib_abs)

    V_sgn = torch.zeros(N, D, device=dev, dtype=dt)
    V_sgn.index_add_(0, nodes_flat, contrib_sgn)

    # Normalize by total participation weight per node
    denom = torch.zeros(N, device=dev, dtype=dt)
    denom.index_add_(0, nodes_flat, w_abs)
    denom = denom.clamp_min(1e-8).unsqueeze(1)
    V_abs = V_abs / denom
    V_sgn = V_sgn / denom

    if l2_normalize:
        V_abs = torch.nn.functional.normalize(V_abs, p=2, dim=1, eps=1e-6)
        V_sgn = torch.nn.functional.normalize(V_sgn, p=2, dim=1, eps=1e-6)

    return V_abs, V_sgn


class MultiViewDGT(nn.Module):
    """
    Differential fusion over multiple edge 'views' with optional portfolio + market fusion.
    Assumes a single homogeneous PyG Data graph (directed edges, duplicated), and
    boolean masks per view over data.edge_index columns.
    """
    def __init__(
        self,
        x_dim: int,
        hidden: int = 128,
        heads: int = 4,
        dropout: float = 0.1,
        view_masks: Dict[str, torch.Tensor] | None = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
        mkt_dim: int = 0,
        use_portfolio: bool = True,
        use_market: bool = True,
        trade_dim: int = 0,
        view_names: Optional[list[str]] = None,
    ):
        super().__init__()
        if TransformerConv is None:
            raise ImportError("torch-geometric is required for MultiViewDGT; install torch-geometric")
        assert view_masks is not None and edge_index is not None, "view_masks and edge_index are required"

        self.use_portfolio = use_portfolio
        self.use_market = use_market and (mkt_dim > 0)
        self.view_names = list(view_names) if (view_names is not None and len(view_names) > 0) else ["struct", "port", "corr_global", "corr_local"]

        # input projection
        self.enc = nn.Sequential(
            nn.Linear(x_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.norm0 = nn.LayerNorm(hidden)

        # per-view convs (two layers for more capacity)
        def _conv():
            # edge_dim=1 so we can feed edge_weight as a 1-dim attribute
            return TransformerConv(hidden,
                                   hidden,
                                   heads=heads,
                                   concat=False,
                                   dropout=dropout,
                                   edge_dim=1,
                                   beta=True)
        self.conv1 = nn.ModuleDict({v: _conv() for v in self.view_names})
        self.conv2 = nn.ModuleDict({v: _conv() for v in self.view_names})

        # learnable scalar gates (one per view) — logit parameterization, then sigmoid to [0,1]
        self.g1_logit = nn.ParameterDict({v: nn.Parameter(torch.zeros(())) for v in self.view_names})
        self.g2_logit = nn.ParameterDict({v: nn.Parameter(torch.zeros(())) for v in self.view_names})

        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

        # Optional learnable gate to softly down‑weight correlation edges (PCC/MI)
        # Initialized at -1.0 so sigmoid≈0.27 initial scaling
        self.corr_gate = nn.Parameter(torch.tensor(-1.0))

        # market encoder
        if self.use_market:
            self.mkt_enc = nn.Sequential(
                nn.Linear(mkt_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden)
            )
        else:
            self.mkt_enc = None

        # trade encoder (optional)
        self.trade_enc = nn.Sequential(
            nn.Linear(trade_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        ) if trade_dim and trade_dim > 0 else None

        # portfolio residual fusion: tiny projection + gate
        self.pf_proj = nn.Linear(2 * hidden, hidden)
        self.pf_gate = nn.Parameter(torch.tensor(0.0))  # start near 0; learnable gate

        # regression head (anchor [+ optional market, trade])
        in_head = hidden
        if self.use_market:
            in_head += hidden
        if self.trade_enc is not None:
            in_head += hidden
        self.head = nn.Sequential(
            nn.Linear(in_head, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

        # store graph + masks (buffers so they move with .to(device))
        self.register_buffer("edge_index_all", edge_index.long(), persistent=False)
        ew = edge_weight if edge_weight is not None else torch.ones(edge_index.size(1), dtype=torch.float32)
        self.register_buffer("edge_weight_all", ew.view(-1, 1).float(), persistent=False)
        for name, mask in (view_masks or {}).items():
            self.register_buffer(f"mask_{name}", mask.bool(), persistent=False)

        # attention capture toggles/state (for TensorBoard stats)
        self._capture_attn: bool = False
        self._attn_stats: dict = {}

    def _run_layer(self, x: torch.Tensor, layer: int) -> torch.Tensor:
        # structural baseline
        ei_s = self.edge_index_all[:, getattr(self, "mask_struct")]
        ew_s = self.edge_weight_all[getattr(self, "mask_struct")]

        convs = self.conv1 if layer == 1 else self.conv2
        glog  = self.g1_logit if layer == 1 else self.g2_logit
        g     = {k: torch.sigmoid(v) for k, v in glog.items()}

        # helper to maybe record attention statistics
        def _maybe_record(view: str, out):
            if not getattr(self, "_capture_attn", False):
                return out
            # When return_attention_weights=True, conv returns (h, (ei, alpha))
            if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], tuple):
                h, att = out
                _ei_att, alpha = att
                mean, std = extract_heads_mean_std(alpha)
                layer_key = f"l{layer}"
                # store under self._attn_stats
                if not hasattr(self, "_attn_stats") or self._attn_stats is None:
                    self._attn_stats = {}
                attn_store_update(self._attn_stats, layer_key, view, mean, std)
                return h
            return out

        # run structural conv
        if getattr(self, "_capture_attn", False):
            out_s = convs["struct"](x, ei_s, edge_attr=ew_s, return_attention_weights=True)
            h_s = _maybe_record("struct", out_s)
        else:
            h_s = convs["struct"](x, ei_s, edge_attr=ew_s)

        def _msg(view: str):
            mask = getattr(self, f"mask_{view}")
            if mask.numel() == 0 or mask.sum() == 0:
                return torch.zeros_like(h_s)
            ei = self.edge_index_all[:, mask]
            ew = self.edge_weight_all[mask]
            # per-view standardization to remove scale mismatch
            # Use population std (unbiased=False) to avoid NaNs for single-edge views
            ew = (ew - ew.mean()) / (ew.std(unbiased=False) + 1e-6)
            # Softly gate correlation-based edges (PCC/MI) to avoid overpowering structural priors
            if view in ("corr_global", "corr_local"):
                gate = torch.sigmoid(self.corr_gate)  # scalar in (0,1)
                ew = ew * gate
            if self._capture_attn:
                out = convs[view](x, ei, edge_attr=ew, return_attention_weights=True)
                return _maybe_record(view, out)
            return convs[view](x, ei, edge_attr=ew)

        h_port = _msg("port")
        h_cg   = _msg("corr_global")
        h_cl   = _msg("corr_local")

        # differential fusion wrt structural with gated contributions in [0,1]
        h = x + g["struct"]*h_s \
              + g["port"]*(h_port - h_s) \
              + g["corr_global"]*(h_cg - h_s) \
              + g["corr_local"]*(h_cl - h_s)
        return h

    def _portfolio_vectors(
        self,
        H: torch.Tensor,
        pf_gid: torch.Tensor,
        port_ctx: Optional[dict] = None,
    ) -> torch.Tensor:
        """Compatibility helper (legacy API): per-sample weighted portfolio sum.
        Returns [B, D] vectors using signed weights over the basket indicated by pf_gid.
        If gid < 0 or context is disabled/absent, returns zeros for that sample.
        This method is not used in the main forward pass anymore; kept for tests/backward-compat.
        Expects port_ctx to provide at least: port_nodes_flat, port_w_signed_flat, port_len.
        Uses port_offsets if available, otherwise infers from cumulative lengths.
        """
        B, D = int(pf_gid.size(0)), int(H.size(1))
        device = H.device
        out = torch.zeros((B, D), device=device, dtype=H.dtype)
        if (not self.use_portfolio) or (port_ctx is None):
            return out
        nodes_flat = port_ctx["port_nodes_flat"].long().to(device)
        w_sgn_flat = port_ctx["port_w_signed_flat"].to(device).to(H.dtype)
        lengths    = port_ctx["port_len"].long().to(device)
        if "port_offsets" in port_ctx:
            offsets = port_ctx["port_offsets"].long().to(device)
        else:
            # infer offsets as exclusive prefix sum
            offsets = torch.zeros_like(lengths)
            if lengths.numel() > 0:
                offsets[1:] = torch.cumsum(lengths[:-1], dim=0)
        for i, gid in enumerate(pf_gid.long().tolist()):
            if gid < 0 or gid >= int(lengths.numel()):
                continue
            o = int(offsets[gid].item()); L = int(lengths[gid].item())
            if L <= 0:
                continue
            idx = nodes_flat[o:o+L]
            w   = w_sgn_flat[o:o+L].view(-1, 1)
            out[i] = (w * H.index_select(0, idx)).sum(dim=0)
        return out


    def forward(
        self,
        x: torch.Tensor,                    # [N, x_dim]
        anchor_idx: torch.Tensor,           # [B]
        market_feat: Optional[torch.Tensor] = None,   # [B, mkt_dim]
        pf_gid: Optional[torch.Tensor] = None,        # [B]
        port_ctx: Optional[dict] = None,
        trade_feat: Optional[torch.Tensor] = None,    # [B, trade_dim]
    ) -> torch.Tensor:
        # encode nodes
        h0 = self.norm0(self.enc(x))

        # layer 1
        h1 = self._run_layer(h0, layer=1)
        h1 = self.norm1(h1)

        # layer 2
        h2 = self._run_layer(h1, layer=2)
        h2 = self.norm2(h2)

        # portfolio residual fusion over node embeddings (independent of batch)
        if self.use_portfolio and (port_ctx is not None):
            V_abs, V_sgn = _portfolio_vectors(h2, port_ctx, l2_normalize=True)  # (N, D), (N, D)
            pf_feat = torch.cat([V_abs, V_sgn], dim=-1)                         # (N, 2D)
            h2 = h2 + torch.sigmoid(self.pf_gate) * self.pf_proj(pf_feat)

        # assemble sample-wise heads
        z_anchor = h2.index_select(0, anchor_idx.long())
        z_list = [z_anchor]

        if self.use_market and (market_feat is not None):
            z_mkt = self.mkt_enc(market_feat)
            z_list.append(z_mkt)

        if (self.trade_enc is not None) and (trade_feat is not None):
            z_trade = self.trade_enc(trade_feat)
            z_list.append(z_trade)

        z = torch.cat(z_list, dim=1)
        yhat = self.head(z).squeeze(-1)   # [B]
        return yhat
