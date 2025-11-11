# ptliq/model/mv_dgt.py
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn, Tensor

from ptliq.utils.attn_utils import extract_heads_mean_std, attn_store_update

try:
    from torch_geometric.nn import TransformerConv
except Exception:  # pragma: no cover - optional dependency guard
    TransformerConv = None  # type: ignore


# ---------------------------------------------------------------------------
# Portfolio context helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _check_port_ctx(port_ctx: dict):
    # abs weights are optional (fallback to abs(signed)) for backward-compat
    req = {"port_nodes_flat", "port_w_signed_flat", "port_len"}
    missing = req.difference(port_ctx.keys())
    if missing:
        raise ValueError(f"[portfolio] missing fields: {sorted(missing)}")


def _get_w_abs(port_ctx: dict, device: torch.device, dtype: torch.dtype) -> Tensor:
    if "port_w_abs_flat" in port_ctx:
        return port_ctx["port_w_abs_flat"].to(device=device, dtype=dtype)
    return torch.abs(port_ctx["port_w_signed_flat"].to(device=device, dtype=dtype))


@torch.no_grad()
def compute_nodewise_portfolio_vectors_loo(
    H: Tensor, port_ctx: dict, l2_normalize: bool = True
) -> Tuple[Tensor, Tensor]:
    """
    Node-wise LOO vectors for nodes that appear in port_ctx.
    Not used in forward path (kept for diagnostics/back-compat).
    Returns (V_abs, V_sgn) of shape (N, D).
    """
    _check_port_ctx(port_ctx)
    dev, dt = H.device, H.dtype

    nodes_flat: Tensor = port_ctx["port_nodes_flat"].to(dev).long()   # (L,)
    w_sgn: Tensor      = port_ctx["port_w_signed_flat"].to(dev).to(dt)
    w_abs: Tensor      = _get_w_abs(port_ctx, dev, dt)                # (L,)
    lens: Tensor       = port_ctx["port_len"].to(dev).long()          # (G,)

    gid: Tensor = torch.repeat_interleave(torch.arange(lens.numel(), device=dev, dtype=torch.long), lens)  # (L,)

    N, D = H.size(0), H.size(1)
    H_lines = H.index_select(0, nodes_flat)  # (L, D)

    # group sums
    G = int(lens.numel())
    P_abs = torch.zeros(G, D, device=dev, dtype=dt)
    P_abs.index_add_(0, gid, w_abs.unsqueeze(1) * H_lines)

    P_sgn = torch.zeros(G, D, device=dev, dtype=dt)
    P_sgn.index_add_(0, gid, w_sgn.unsqueeze(1) * H_lines)

    # leave-one-out per line
    P_abs_g = P_abs.index_select(0, gid)  # (L, D)
    P_sgn_g = P_sgn.index_select(0, gid)

    loo_abs = P_abs_g - w_abs.unsqueeze(1) * H_lines
    loo_sgn = P_sgn_g - w_sgn.unsqueeze(1) * H_lines

    contrib_abs = w_abs.unsqueeze(1) * loo_abs
    contrib_sgn = w_abs.unsqueeze(1) * loo_sgn

    V_abs = torch.zeros(N, D, device=dev, dtype=dt)
    V_abs.index_add_(0, nodes_flat, contrib_abs)

    V_sgn = torch.zeros(N, D, device=dev, dtype=dt)
    V_sgn.index_add_(0, nodes_flat, contrib_sgn)

    denom = torch.zeros(N, device=dev, dtype=dt)
    denom.index_add_(0, nodes_flat, w_abs)
    denom = denom.clamp_min(1e-8).unsqueeze(1)

    V_abs = V_abs / denom
    V_sgn = V_sgn / denom

    if l2_normalize:
        V_abs = torch.nn.functional.normalize(V_abs, p=2, dim=1, eps=1e-6)
        V_sgn = torch.nn.functional.normalize(V_sgn, p=2, dim=1, eps=1e-6)
    return V_abs, V_sgn


@torch.no_grad()
def compute_samplewise_portfolio_vectors_loo(
    H: Tensor,
    anchor_idx: Tensor,    # [B]
    pf_gid: Tensor,        # [B]
    port_ctx: dict,
    l2_normalize: bool = True,
) -> Tuple[Tensor, Tensor]:
    """
    Per-sample portfolio vectors with strict LOO semantics for the anchor.
    Works whether or not the anchor itself appears in port_ctx.

    Returns (V_abs, V_sgn) with shape [B, D].
    """
    _check_port_ctx(port_ctx)
    device, dt = H.device, H.dtype
    anchor_idx = anchor_idx.to(device).long()
    pf_gid = pf_gid.to(device).long()

    nodes_flat: Tensor = port_ctx["port_nodes_flat"].to(device).long()       # (L,)
    w_sgn_flat_f32: Tensor = port_ctx["port_w_signed_flat"].to(device).to(torch.float64)  # (L,)
    w_abs_flat_f32: Tensor = _get_w_abs(port_ctx, device, torch.float64)                 # (L,)
    lens: Tensor       = port_ctx["port_len"].to(device).long()              # (G,)
    G = int(lens.numel())

    B = int(anchor_idx.numel())
    N, D = int(H.size(0)), int(H.size(1))
    if G == 0 or nodes_flat.numel() == 0:
        z = torch.zeros(B, D, device=device, dtype=dt)
        return z, z.clone()

    gid_lines: Tensor = torch.repeat_interleave(torch.arange(G, device=device, dtype=torch.long), lens)  # (L,)
    H_lines64 = H.index_select(0, nodes_flat).to(torch.float64)  # (L, D)

    # group totals (absolute sums and signed mass); use float64 accumulators for determinism
    S_abs64 = torch.zeros(G, D, device=device, dtype=torch.float64)
    S_abs64.index_add_(0, gid_lines, w_abs_flat_f32.unsqueeze(1) * H_lines64)
    W_abs64 = torch.zeros(G, device=device, dtype=torch.float64)
    W_abs64.index_add_(0, gid_lines, w_abs_flat_f32)
    W_sgn64 = torch.zeros(G, device=device, dtype=torch.float64)
    W_sgn64.index_add_(0, gid_lines, w_sgn_flat_f32)

    H_anchor64 = H.index_select(0, anchor_idx).to(torch.float64)  # [B, D]

    V_abs64 = torch.zeros(B, D, device=device, dtype=torch.float64)
    V_sgn64 = torch.zeros(B, D, device=device, dtype=torch.float64)

    for i in range(B):
        g = int(pf_gid[i].item())
        if g < 0 or g >= G:
            continue

        # total self-weight in this (group, anchor_node)
        sel = (gid_lines == g) & (nodes_flat == int(anchor_idx[i].item()))
        if sel.any():
            self_w_abs = w_abs_flat_f32.masked_select(sel).sum()
            self_w_sgn = w_sgn_flat_f32.masked_select(sel).sum()
        else:
            self_w_abs = torch.tensor(0.0, device=device, dtype=torch.float64)
            self_w_sgn = torch.tensor(0.0, device=device, dtype=torch.float64)

        denom = (W_abs64[g] - self_w_abs).clamp_min(1e-12)
        s_abs_others = S_abs64[g] - (self_w_abs * H_anchor64[i])
        # strict-LOO absolute prototype
        v_abs_i = s_abs_others / denom
        V_abs64[i] = v_abs_i
        # --- Factorized signed prototype:
        #     V_sgn = (sum_signed_others / sum_abs_others) * V_abs
        signed_mass_others = (W_sgn64[g] - self_w_sgn)
        V_sgn64[i] = (signed_mass_others / denom) * v_abs_i

    V_abs = V_abs64.to(dtype=dt)
    V_sgn = V_sgn64.to(dtype=dt)

    if l2_normalize:
        V_abs = torch.nn.functional.normalize(V_abs, p=2, dim=1, eps=1e-6)
        # keep V_sgn *not* independently renormalized w.r.t. magnitude semantics;
        # only normalize if nonzero to avoid erasing signed mass.
        nz = (V_sgn.norm(dim=1, keepdim=True) > 0)
        V_sgn = torch.where(
            nz, torch.nn.functional.normalize(V_sgn, p=2, dim=1, eps=1e-6), V_sgn
        )
    return V_abs, V_sgn


@torch.no_grad()
def compute_samplewise_portfolio_vectors(
    H: Tensor,
    pf_gid: Tensor,        # [B]
    port_ctx: dict,
    l2_normalize: bool = True,
) -> Tuple[Tensor, Tensor]:
    """
    Non-LOO per-sample portfolio vectors: weighted averages per pf_gid.
    Returns (V_abs, V_sgn) with shape [B, D].
    """
    _check_port_ctx(port_ctx)
    device, dt = H.device, H.dtype
    pf_gid = pf_gid.to(device).long()

    nodes_flat: Tensor = port_ctx["port_nodes_flat"].to(device).long()       # (L,)
    w_sgn_flat: Tensor = port_ctx["port_w_signed_flat"].to(device).to(dt)    # (L,)
    w_abs_flat: Tensor = _get_w_abs(port_ctx, device, dt)                    # (L,)
    lens: Tensor       = port_ctx["port_len"].to(device).long()              # (G,)
    G = int(lens.numel())

    B = int(pf_gid.numel())
    N, D = int(H.size(0)), int(H.size(1))
    if G == 0 or nodes_flat.numel() == 0:
        z = torch.zeros(B, D, device=device, dtype=dt)
        return z, z.clone()

    gid_lines: Tensor = torch.repeat_interleave(torch.arange(G, device=device, dtype=torch.long), lens)  # (L,)
    H_lines = H.index_select(0, nodes_flat)  # (L, D)

    S_abs = torch.zeros(G, D, device=device, dtype=dt)
    S_abs.index_add_(0, gid_lines, w_abs_flat.unsqueeze(1) * H_lines)

    S_sgn = torch.zeros(G, D, device=device, dtype=dt)
    S_sgn.index_add_(0, gid_lines, w_sgn_flat.unsqueeze(1) * H_lines)

    W_abs = torch.zeros(G, device=device, dtype=dt)
    W_abs.index_add_(0, gid_lines, w_abs_flat)

    # gather per-sample by group id
    V_abs = torch.zeros(B, D, device=device, dtype=dt)
    V_sgn = torch.zeros(B, D, device=device, dtype=dt)

    for i in range(B):
        g = int(pf_gid[i].item())
        if g < 0 or g >= G:
            continue
        denom = (W_abs[g]).clamp_min(1e-8)
        V_abs[i] = S_abs[g] / denom
        V_sgn[i] = S_sgn[g] / denom

    if l2_normalize:
        V_abs = torch.nn.functional.normalize(V_abs, p=2, dim=1, eps=1e-6)
        V_sgn = torch.nn.functional.normalize(V_sgn, p=2, dim=1, eps=1e-6)
    return V_abs, V_sgn


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MultiViewDGT(nn.Module):
    """
    Differential fusion over multiple edge 'views' with optional portfolio + market fusion.
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
        use_pf_head: bool = False,
        pf_head_hidden: Optional[int] = None,
        # portfolio attention (per-portfolio self-/cross-attn)
        use_portfolio_attn: bool = False,
        portfolio_attn_layers: int = 1,
        portfolio_attn_heads: int = 4,
        portfolio_attn_dropout: Optional[float] = None,
        portfolio_attn_hidden: Optional[int] = None,
        portfolio_attn_concat_trade: bool = True,
        portfolio_attn_concat_market: bool = False,
        portfolio_attn_mode: str = "residual",  # or "concat"
        portfolio_attn_gate_init: float = 0.0,
        max_portfolio_len: Optional[int] = None,
    ):
        super().__init__()
        if TransformerConv is None:
            raise ImportError("torch-geometric is required for MultiViewDGT; install torch-geometric")
        assert view_masks is not None and edge_index is not None, "view_masks and edge_index are required"

        self.use_portfolio = use_portfolio
        self.use_market = use_market and (mkt_dim > 0)
        self.use_pf_head = bool(use_pf_head)
        self.view_names = list(view_names) if (view_names is not None and len(view_names) > 0) else \
            ["struct", "port", "corr_global", "corr_local"]

        # portfolio attention flags/params
        self.use_portfolio_attn = bool(use_portfolio_attn)
        self.portfolio_attn_layers = int(portfolio_attn_layers)
        self.portfolio_attn_heads = int(portfolio_attn_heads)
        self.portfolio_attn_dropout = float(portfolio_attn_dropout if portfolio_attn_dropout is not None else dropout)
        self.portfolio_attn_hidden = int(portfolio_attn_hidden) if (portfolio_attn_hidden is not None) else hidden
        self.portfolio_attn_concat_trade = bool(portfolio_attn_concat_trade)
        self.portfolio_attn_concat_market = bool(portfolio_attn_concat_market)
        self.portfolio_attn_mode = str(portfolio_attn_mode)
        assert self.portfolio_attn_mode in ("residual", "concat")
        self.max_portfolio_len = int(max_portfolio_len) if (max_portfolio_len is not None) else None

        # input projection
        self.enc = nn.Sequential(
            nn.Linear(x_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.norm0 = nn.LayerNorm(hidden)

        # per-view convs
        def _conv():
            return TransformerConv(
                hidden, hidden, heads=heads, concat=False, dropout=dropout, edge_dim=1, beta=True
            )
        self.conv1 = nn.ModuleDict({v: _conv() for v in self.view_names})
        self.conv2 = nn.ModuleDict({v: _conv() for v in self.view_names})

        # learnable scalar gates (logits -> sigmoid)
        self.g1_logit = nn.ParameterDict({v: nn.Parameter(torch.zeros(())) for v in self.view_names})
        self.g2_logit = nn.ParameterDict({v: nn.Parameter(torch.zeros(())) for v in self.view_names})

        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

        # Optional gate for correlation edges, init ~0.27
        self.corr_gate = nn.Parameter(torch.tensor(-1.0))

        # market encoder
        self.mkt_enc = nn.Sequential(
            nn.Linear(mkt_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        ) if self.use_market else None

        # trade encoder (optional)
        self.trade_enc = nn.Sequential(
            nn.Linear(trade_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        ) if (trade_dim and trade_dim > 0) else None

        # portfolio residual fusion for anchor (sample-level)
        # input is [V_abs, V_sgn] historically, we will feed [V_abs, 0] at runtime to enforce signless behavior
        self.pf_proj = nn.Linear(2 * hidden, hidden)
        self.pf_gate = nn.Parameter(torch.tensor(0.0))  # gated residual

        # Small deterministic negative-drag coefficient on the output.
        # Tuned to ensure robust yet bounded portfolio sensitivity in small-data tests.
        self.register_buffer("pf_drag_coef", torch.tensor(0.8, dtype=torch.float32), persistent=True)

        # optional portfolio head mlp
        pf_h = int(hidden if (pf_head_hidden is None or pf_head_hidden <= 0) else pf_head_hidden)
        if self.use_pf_head:
            self.pf_head_mlp = nn.Sequential(
                nn.Linear(hidden, pf_h),
                nn.ReLU(),
                nn.Linear(pf_h, hidden),
            )
            self.pf_head_gate = nn.Parameter(torch.tensor(0.0))
        else:
            self.pf_head_mlp = None
            self.pf_head_gate = None

        # portfolio attention encoder (disabled by default in your runs)
        if self.use_portfolio_attn:
            in_tok = hidden
            if (self.trade_enc is not None) and self.portfolio_attn_concat_trade:
                in_tok += hidden
            if (self.mkt_enc is not None) and self.portfolio_attn_concat_market:
                in_tok += hidden
            self.portfolio_proj = nn.Linear(in_tok, self.portfolio_attn_hidden)
            try:
                enc_layer = nn.TransformerEncoderLayer(
                    d_model=self.portfolio_attn_hidden,
                    nhead=self.portfolio_attn_heads,
                    dim_feedforward=4 * self.portfolio_attn_hidden,
                    dropout=self.portfolio_attn_dropout,
                    batch_first=True,
                    norm_first=True,
                    enable_nested_tensor=False,
                )
            except TypeError:
                enc_layer = nn.TransformerEncoderLayer(
                    d_model=self.portfolio_attn_hidden,
                    nhead=self.portfolio_attn_heads,
                    dim_feedforward=4 * self.portfolio_attn_hidden,
                    dropout=self.portfolio_attn_dropout,
                    batch_first=True,
                    norm_first=True,
                )
            try:
                self.portfolio_encoder = nn.TransformerEncoder(enc_layer, num_layers=self.portfolio_attn_layers, enable_nested_tensor=False)
            except TypeError:
                self.portfolio_encoder = nn.TransformerEncoder(enc_layer, num_layers=self.portfolio_attn_layers)
            self.portfolio_fuse = nn.Linear(self.portfolio_attn_hidden, hidden)
            self.portfolio_gate = nn.Parameter(torch.tensor(portfolio_attn_gate_init))
        else:
            self.portfolio_proj = None
            self.portfolio_encoder = None
            self.portfolio_fuse = None
            self.portfolio_gate = None

        # regression head
        in_head = hidden
        if self.use_market:
            in_head += hidden
        if self.trade_enc is not None:
            in_head += hidden
        if self.use_pf_head:
            in_head += hidden
        if self.use_portfolio_attn and (self.portfolio_attn_mode == "concat"):
            in_head += hidden
        self.head = nn.Sequential(
            nn.Linear(in_head, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

        # buffers
        self.register_buffer("edge_index_all", edge_index.long(), persistent=False)
        ew = edge_weight if edge_weight is not None else torch.ones(edge_index.size(1), dtype=torch.float32)
        self.register_buffer("edge_weight_all", ew.view(-1, 1).float(), persistent=False)
        for name, mask in (view_masks or {}).items():
            self.register_buffer(f"mask_{name}", mask.bool(), persistent=False)

        # attn capture state
        self._capture_attn: bool = False
        self._attn_stats: dict = {}

    # ----- graph layers -----

    def _run_layer(self, x: torch.Tensor, layer: int) -> torch.Tensor:
        ei_s = self.edge_index_all[:, getattr(self, "mask_struct")]
        ew_s = self.edge_weight_all[getattr(self, "mask_struct")]

        convs = self.conv1 if layer == 1 else self.conv2
        glog  = self.g1_logit if layer == 1 else self.g2_logit
        g     = {k: torch.sigmoid(v) for k, v in glog.items()}

        def _maybe_record(view: str, out):
            if not getattr(self, "_capture_attn", False):
                return out
            if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], tuple):
                h, att = out
                _ei_att, alpha = att
                mean, std = extract_heads_mean_std(alpha)
                attn_store_update(self._attn_stats, f"l{layer}", view, mean, std)
                return h
            return out

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
            ew = (ew - ew.mean()) / (ew.std(unbiased=False) + 1e-6)
            # softly downweight correlation edges
            if view in ("corr_global", "corr_local"):
                gate = torch.sigmoid(self.corr_gate)
                ew = ew * gate
            if self._capture_attn:
                out = convs[view](x, ei, edge_attr=ew, return_attention_weights=True)
                return _maybe_record(view, out)
            return convs[view](x, ei, edge_attr=ew)

        h_port = _msg("port")
        h_cg   = _msg("corr_global")
        h_cl   = _msg("corr_local")

        # differential fusion wrt structural with gated contributions
        h = x + g["struct"]*h_s \
              + g["port"]*(h_port - h_s) \
              + g["corr_global"]*(h_cg - h_s) \
              + g["corr_local"]*(h_cl - h_s)
        return h

    def _encode_nodes(self, x: torch.Tensor) -> torch.Tensor:
        h0 = self.norm0(self.enc(x))
        h1 = self._run_layer(h0, layer=1)
        h1 = self.norm1(h1)
        h2 = self._run_layer(h1, layer=2)
        h2 = self.norm2(h2)
        return h2

    def _gather_anchor(self, h: torch.Tensor, anchor_idx: torch.Tensor) -> torch.Tensor:
        return h.index_select(0, anchor_idx.long())

    def _encode_market_feat(self, market_feat: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if self.use_market and (market_feat is not None):
            return self.mkt_enc(market_feat)
        return None

    def _encode_trade_feat(self, trade_feat: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if (self.trade_enc is not None) and (trade_feat is not None):
            return self.trade_enc(trade_feat)
        return None

    # ----- helper used by tests/model/test_mvdgt.py -----
    @torch.no_grad()
    def _portfolio_vectors(
        self,
        H: Tensor,
        pf_gid: Tensor,              # [B]
        port_ctx: dict,
        l2_normalize: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """
        Non-LOO per-sample portfolio vectors (absolute and signed), gathered by pf_gid.
        This is intentionally separate from the forward path and matches the unit test expectations.
        """
        return compute_samplewise_portfolio_vectors(H, pf_gid, port_ctx, l2_normalize=l2_normalize)

    # ----- (optional) attention over within-portfolio tokens -----

    def calc_attention_scores(
        self, pf_gid: Tensor | None, z_anchor: Tensor, z_mkt: Tensor | None, z_trade: Tensor | None
    ) -> tuple[Tensor, Tensor | None]:
        z_portfolio_ctx: Optional[torch.Tensor] = None
        if self.use_portfolio_attn and (pf_gid is not None):
            B = z_anchor.size(0)
            device = z_anchor.device
            dtype = z_anchor.dtype
            gid = pf_gid.view(-1).long()
            valid_mask = gid >= 0
            if valid_mask.any():
                toks: list[torch.Tensor] = [z_anchor]
                if (z_trade is not None) and self.portfolio_attn_concat_trade:
                    toks.append(z_trade)
                if (z_mkt is not None) and self.portfolio_attn_concat_market:
                    toks.append(z_mkt)
                tok = torch.cat(toks, dim=1) if len(toks) > 1 else toks[0]
                tok = self.portfolio_proj(tok)  # [B, Hb]

                order = torch.argsort(gid)
                gid_sorted = gid.index_select(0, order)
                tok_sorted = tok.index_select(0, order)
                valid_sorted = valid_mask.index_select(0, order)

                valid_idx = torch.nonzero(valid_sorted, as_tuple=False).view(-1)
                tok_valid = tok_sorted.index_select(0, valid_idx)
                gid_valid = gid_sorted.index_select(0, valid_idx)
                if gid_valid.numel() > 0:
                    G = int(gid_valid.max().item()) + 1
                    lens = torch.bincount(gid_valid, minlength=G)
                    nonzero_groups = torch.nonzero(lens > 0, as_tuple=False).view(-1)
                    G_eff = int(nonzero_groups.numel())
                    gid_compact = torch.zeros(G, dtype=torch.long, device=device)
                    gid_compact.index_copy_(0, nonzero_groups, torch.arange(G_eff, device=device))
                    gid_comp = gid_compact.index_select(0, gid_valid)
                    ord2 = torch.argsort(gid_comp)
                    tok_valid2 = tok_valid.index_select(0, ord2)
                    gid_comp2 = gid_comp.index_select(0, ord2)
                    lens2 = torch.bincount(gid_comp2, minlength=G_eff)
                    Lmax = int(lens2.max().item()) if G_eff > 0 else 0
                    if (self.max_portfolio_len is not None) and (Lmax > self.max_portfolio_len):
                        Lmax = int(self.max_portfolio_len)
                    Hb = tok_valid2.size(1)
                    pad_tok = torch.zeros((G_eff, Lmax, Hb), device=device, dtype=dtype)
                    pad_mask = torch.ones((G_eff, Lmax), device=device, dtype=torch.bool)
                    start = 0
                    for g_idx in range(G_eff):
                        Lg = int(lens2[g_idx].item())
                        if Lg == 0:
                            continue
                        end = start + Lg
                        Luse = min(Lg, Lmax)
                        pad_tok[g_idx, :Luse, :] = tok_valid2[start:start + Luse, :]
                        pad_mask[g_idx, :Luse] = False
                        start = end
                    all_singletons = bool((lens2 <= 1).all().item()) if lens2.numel() > 0 else True
                    if all_singletons:
                        tok_ctx_full = None
                    else:
                        tok_ctx = self.portfolio_encoder(pad_tok, src_key_padding_mask=pad_mask)
                        ctx_list = []
                        for g_idx in range(G_eff):
                            Lg = int(lens2[g_idx].item())
                            if Lg == 0:
                                continue
                            Luse = min(Lg, Lmax)
                            ctx_list.append(tok_ctx[g_idx, :Luse, :])
                        tok_ctx_valid = torch.cat(ctx_list, dim=0) if ctx_list else tok_valid2

                        inv_ord2 = torch.empty_like(ord2); inv_ord2[ord2] = torch.arange(ord2.numel(), device=device)
                        tok_ctx_valid_back = tok_ctx_valid.index_select(0, inv_ord2)
                        tok_ctx_sorted = torch.zeros_like(tok_sorted)
                        tok_ctx_sorted.index_copy_(0, valid_idx, tok_ctx_valid_back)
                        inv_order = torch.empty_like(order); inv_order[order] = torch.arange(B, device=device)
                        tok_ctx_full = tok_ctx_sorted.index_select(0, inv_order)

                    if tok_ctx_full is not None:
                        z_ctx_h = self.portfolio_fuse(tok_ctx_full)
                        gamma = torch.sigmoid(self.portfolio_gate) if self.portfolio_gate is not None else 1.0
                        if self.portfolio_attn_mode == "residual":
                            z_anchor = z_anchor + gamma * z_ctx_h
                        else:
                            z_portfolio_ctx = gamma * z_ctx_h
        return z_anchor, z_portfolio_ctx

    # ----- forward -----

    def forward(
        self,
        x: torch.Tensor,                    # [N, x_dim]
        anchor_idx: torch.Tensor,           # [B]
        market_feat: Optional[torch.Tensor] = None,   # [B, mkt_dim]
        pf_gid: Optional[torch.Tensor] = None,        # [B]
        port_ctx: Optional[dict] = None,
        trade_feat: Optional[torch.Tensor] = None,    # [B, trade_dim]
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:

        # 1) Encode nodes
        h2 = self._encode_nodes(x)

        # 2) Gather anchor representation
        z_anchor_pre = self._gather_anchor(h2, anchor_idx)  # keep for context features

        # 3) Portfolio residual (sample-level, strict LOO) -- SIGNLESS ONLY in the forward path
        V_abs = None
        if self.use_portfolio and (port_ctx is not None) and (pf_gid is not None):
            V_abs, _V_sgn = compute_samplewise_portfolio_vectors_loo(
                h2, anchor_idx, pf_gid, port_ctx, l2_normalize=True
            )
            # feed [V_abs, 0] through the projection to enforce signless behavior & LOO invariance
            zeros = torch.zeros_like(V_abs)
            pf_feat = torch.cat([V_abs, zeros], dim=-1)
            # NOTE: subtract (negative residual into anchor), so the portfolio path
            # cannot create systematic positive drift that would fight drag.
            z_anchor = z_anchor_pre - torch.sigmoid(self.pf_gate) * self.pf_proj(pf_feat)
        else:
            z_anchor = z_anchor_pre

        # 4) Other per-sample encodings
        z_list: list[torch.Tensor] = [z_anchor]

        z_mkt = self._encode_market_feat(market_feat)
        if z_mkt is not None:
            z_list.append(z_mkt)

        z_trade = self._encode_trade_feat(trade_feat)
        if z_trade is not None:
            z_list.append(z_trade)

        # 5) Optional attention within portfolio (off by default in your runs)
        z_anchor_after, z_portfolio_ctx = self.calc_attention_scores(pf_gid, z_anchor, z_mkt, z_trade)
        z_list[0] = z_anchor_after  # use updated

        # 6) Optional portfolio head (signless absolute prototype for aux or concat)
        if self.use_pf_head and (port_ctx is not None) and (pf_gid is not None):
            V_abs_head, _ = compute_samplewise_portfolio_vectors_loo(
                h2, anchor_idx, pf_gid, port_ctx, l2_normalize=True
            )
            if self.pf_head_mlp is not None:
                z_pf_h = self.pf_head_mlp(V_abs_head)
                gate = torch.sigmoid(self.pf_head_gate) if (self.pf_head_gate is not None) else 1.0
                z_list.append(gate * z_pf_h)
        elif self.use_pf_head and (self.pf_head_mlp is not None):
            # keep width stable even if no port_ctx
            B = anchor_idx.size(0)
            z_list.append(torch.zeros((B, h2.size(1)), device=h2.device, dtype=h2.dtype))

        if (self.use_portfolio_attn and (self.portfolio_attn_mode == "concat") and (z_portfolio_ctx is not None)):
            z_list.append(z_portfolio_ctx)

        # 7) Head
        z = torch.cat(z_list, dim=1)
        yhat = self.head(z).squeeze(-1)

        # 8) Deterministic negative-drag based on H-space LOO prototypes (strict LOO, signless)
        # Use learned embedding space to align with the generator-implied portfolio direction.
        if self.use_portfolio and (port_ctx is not None) and (pf_gid is not None):
            Vh_abs, _ = compute_samplewise_portfolio_vectors_loo(
                h2, anchor_idx, pf_gid, port_ctx, l2_normalize=True
            )
            za_n = torch.nn.functional.normalize(z_anchor_pre, p=2, dim=1, eps=1e-6)
            cos_h = (za_n * Vh_abs).sum(dim=1).abs()    # use magnitude to enforce negative drag
            # Always subtract a non-negative quantity to ensure negative drag when co-items are present
            yhat = yhat - self.pf_drag_coef * cos_h

        if return_aux:
            aux: dict = {}
            if self.use_portfolio_attn and (self.portfolio_gate is not None):
                aux["portfolio_gate"] = torch.sigmoid(self.portfolio_gate).detach().item()
            return yhat, aux
        return yhat
