# mv_dgt.py
from __future__ import annotations
from typing import Dict, Optional, Any
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
      - V_abs: average portfolio intensity vector per node (abs weights)
      - V_sgn: average portfolio direction vector per node (signed weights)
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

    # portfolio-level sums
    P_abs = torch.zeros(G, D, device=dev, dtype=dt)
    P_abs.index_add_(0, gid, w_abs.unsqueeze(1) * H_lines)

    P_sgn = torch.zeros(G, D, device=dev, dtype=dt)
    P_sgn.index_add_(0, gid, w_sgn.unsqueeze(1) * H_lines)

    # Leave-one-out portfolio vectors per line
    P_abs_g = P_abs.index_select(0, gid)  # (L, D)
    P_sgn_g = P_sgn.index_select(0, gid)  # (L, D)

    loo_abs = P_abs_g - w_abs.unsqueeze(1) * H_lines
    loo_sgn = P_sgn_g - w_sgn.unsqueeze(1) * H_lines

    # Contribution to the node from each portfolio membership, weighted by own |w_abs|
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
        self.view_names = list(view_names) if (view_names is not None and len(view_names) > 0) else ["struct", "port", "corr_global", "corr_local"]

        # portfolio attention flags/params
        self.use_portfolio_attn = bool(use_portfolio_attn)
        self.portfolio_attn_layers = int(portfolio_attn_layers)
        self.portfolio_attn_heads = int(portfolio_attn_heads)
        self.portfolio_attn_dropout = float(portfolio_attn_dropout if portfolio_attn_dropout is not None else dropout)
        self.portfolio_attn_hidden = int(portfolio_attn_hidden) if (portfolio_attn_hidden is not None) else hidden
        self.portfolio_attn_concat_trade = bool(portfolio_attn_concat_trade)
        self.portfolio_attn_concat_market = bool(portfolio_attn_concat_market)
        self.portfolio_attn_mode = str(portfolio_attn_mode)
        assert self.portfolio_attn_mode in ("residual", "concat"), "portfolio_mode must be 'residual' or 'concat'"
        self.max_portfolio_len = int(max_portfolio_len) if (max_portfolio_len is not None) else None

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

        # optional portfolio head mlp (per-sample portfolio embedding)
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

        # portfolio attention encoder (per-portfolio self-/cross-attn)
        if self.use_portfolio_attn:
            in_tok = hidden
            if (self.trade_enc is not None) and self.portfolio_attn_concat_trade:
                in_tok += hidden
            if (self.mkt_enc is not None) and self.portfolio_attn_concat_market:
                in_tok += hidden
            self.portfolio_proj = nn.Linear(in_tok, self.portfolio_attn_hidden)
            # Construct Transformer encoder layer. Newer PyTorch versions support
            # enable_nested_tensor=False to suppress nested-tensor warnings when using
            # key padding masks. Fall back silently on older versions.
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
            # Build encoder; in newer PyTorch versions, pass enable_nested_tensor=False to avoid warnings
            try:
                self.portfolio_encoder = nn.TransformerEncoder(enc_layer, num_layers=self.portfolio_attn_layers, enable_nested_tensor=False)
            except TypeError:
                self.portfolio_encoder = nn.TransformerEncoder(enc_layer, num_layers=self.portfolio_attn_layers)
            # fuse contextualized token back to hidden
            self.portfolio_fuse = nn.Linear(self.portfolio_attn_hidden, hidden)
            self.portfolio_gate = nn.Parameter(torch.tensor(portfolio_attn_gate_init))
        else:
            self.portfolio_proj = None
            self.portfolio_encoder = None
            self.portfolio_fuse = None
            self.portfolio_gate = None

        # regression head (anchor [+ optional market, trade, pf_head, portfolio_ctx])
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
        Returns [B, D] vectors using signed weights over the portfolio indicated by pf_gid.
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


    def _encode_nodes(self, x: torch.Tensor) -> torch.Tensor:
        """Project raw node features and run the two graph layers with normalization."""
        h0 = self.norm0(self.enc(x))
        h1 = self._run_layer(h0, layer=1)
        h1 = self.norm1(h1)
        h2 = self._run_layer(h1, layer=2)
        h2 = self.norm2(h2)
        return h2

    def _apply_portfolio_residual(self, h: torch.Tensor, port_ctx: Optional[dict]) -> torch.Tensor:
        """Apply portfolio residual fusion over node embeddings when context is provided."""
        if self.use_portfolio and (port_ctx is not None):
            V_abs, V_sgn = _portfolio_vectors(h, port_ctx, l2_normalize=True)
            pf_feat = torch.cat([V_abs, V_sgn], dim=-1)
            h = h + torch.sigmoid(self.pf_gate) * self.pf_proj(pf_feat)
        return h

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

    def _compute_pf_head(self, h: torch.Tensor, pf_gid: Optional[torch.Tensor], port_ctx: Optional[dict], return_aux: bool) -> tuple[list[torch.Tensor], dict, Optional[torch.Tensor]]:
        z_list_add: list[torch.Tensor] = []
        aux: dict = {}
        z_pf_feat: Optional[torch.Tensor] = None
        if (pf_gid is not None) and (port_ctx is not None):
            # per-sample portfolio vector (permutation-invariant)
            z_pf_feat = self._portfolio_vectors(h, pf_gid, port_ctx)
            if return_aux:
                aux["z_pf"] = z_pf_feat
            if self.use_pf_head and (self.pf_head_mlp is not None):
                z_pf_h = self.pf_head_mlp(z_pf_feat)
                gate = torch.sigmoid(self.pf_head_gate) if (self.pf_head_gate is not None) else 1.0
                z_list_add.append(gate * z_pf_h)
        return z_list_add, aux, z_pf_feat

    def _append_pf_placeholder_if_needed(self, z_list: list[torch.Tensor], h: torch.Tensor, anchor_idx: torch.Tensor, z_pf_feat: Optional[torch.Tensor]) -> None:
        """If pf head is enabled but z_pf was not appended, add zeros to keep head width constant."""
        if self.use_pf_head and (self.pf_head_mlp is not None):
            appended = isinstance(z_pf_feat, torch.Tensor)
            if not appended:
                B = anchor_idx.size(0)
                z_list.append(torch.zeros((B, h.size(1)), device=h.device, dtype=h.dtype))

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
        # 1) Encode nodes through graph layers
        h2 = self._encode_nodes(x)
        # 2) Portfolio residual fusion (node-level)
        h2 = self._apply_portfolio_residual(h2, port_ctx)
        # 3) Assemble sample-wise heads
        z_anchor = self._gather_anchor(h2, anchor_idx)
        z_list: list[torch.Tensor] = [z_anchor]

        z_mkt = self._encode_market_feat(market_feat)
        if z_mkt is not None:
            z_list.append(z_mkt)

        z_trade = self._encode_trade_feat(trade_feat)
        if z_trade is not None:
            z_list.append(z_trade)

        # 4) portfolio self-/cross-attention over items within each portfolio
        z_anchor_prev = z_anchor
        z_anchor, z_portfolio_ctx = self.calc_attention_scores(pf_gid, z_anchor, z_mkt, z_trade)

        # Ensure head uses updated z_anchor when portfolio residual applied
        z_list[0] = z_anchor

        # Track portfolio delta norm for observability
        portfolio_delta_norm = (z_anchor - z_anchor_prev).norm(dim=1).mean() if self.use_portfolio_attn else None

        add_pf, aux, z_pf_feat = self._compute_pf_head(h2, pf_gid, port_ctx, return_aux)
        if add_pf:
            z_list.extend(add_pf)
        # 5) Keep head input width constant if pf head enabled but unavailable
        self._append_pf_placeholder_if_needed(z_list, h2, anchor_idx, z_pf_feat)

        if (self.use_portfolio_attn and (self.portfolio_attn_mode == "concat") and (z_portfolio_ctx is not None)):
            z_list.append(z_portfolio_ctx)

        z = torch.cat(z_list, dim=1)
        yhat = self.head(z).squeeze(-1)
        if return_aux:
            if self.use_portfolio_attn:
                aux["portfolio_gate"] = torch.sigmoid(self.portfolio_gate).detach().item() if (self.portfolio_gate is not None) else 0.0
                if portfolio_delta_norm is not None:
                    try:
                        aux["portfolio_delta_norm"] = float(portfolio_delta_norm.detach().item())
                    except Exception:
                        pass
            return yhat, aux
        return yhat

    def calc_attention_scores(self, 
                              pf_gid: Tensor | None,
                              z_anchor: Tensor, 
                              z_mkt: Tensor | None, 
                              z_trade: Tensor | None) -> \
    tuple[Tensor | Any, Tensor | None]:
        z_portfolio_ctx: Optional[torch.Tensor] = None
        if self.use_portfolio_attn and (pf_gid is not None):
            B = z_anchor.size(0)
            device = z_anchor.device
            dtype = z_anchor.dtype
            # valid samples are those with gid >= 0
            gid = pf_gid.view(-1).long()
            valid_mask = gid >= 0
            if valid_mask.any():
                # Optionally cap extremely large portfolios by top-|weight| selection at scorer side; here we just guard memory
                # Build token features to feed encoder
                toks: list[torch.Tensor] = [z_anchor]
                if (z_trade is not None) and self.portfolio_attn_concat_trade:
                    toks.append(z_trade)
                if (z_mkt is not None) and self.portfolio_attn_concat_market:
                    toks.append(z_mkt)
                tok = torch.cat(toks, dim=1) if len(toks) > 1 else toks[0]
                tok = self.portfolio_proj(tok)  # [B, Hb]

                # Sort by gid to make groups contiguous
                order = torch.argsort(gid)
                gid_sorted = gid.index_select(0, order)
                tok_sorted = tok.index_select(0, order)
                valid_sorted = valid_mask.index_select(0, order)

                # Compute group lengths (only for valid gids)
                # Identify segment boundaries where gid changes, excluding invalid entries
                valid_idx = torch.nonzero(valid_sorted, as_tuple=False).view(-1)
                tok_valid = tok_sorted.index_select(0, valid_idx)
                gid_valid = gid_sorted.index_select(0, valid_idx)
                if gid_valid.numel() > 0:
                    # bincount up to max gid + 1
                    G = int(gid_valid.max().item()) + 1
                    lens = torch.bincount(gid_valid, minlength=G)
                    # Remove empty groups that might exist between 0..max
                    nonzero_groups = torch.nonzero(lens > 0, as_tuple=False).view(-1)
                    G_eff = int(nonzero_groups.numel())
                    # Map original gid → compact 0..G_eff-1
                    gid_compact = torch.zeros(G, dtype=torch.long, device=device)
                    gid_compact.index_copy_(0, nonzero_groups, torch.arange(G_eff, device=device))
                    gid_comp = gid_compact.index_select(0, gid_valid)
                    # Reorder by compact gid to assemble portfolios contiguously (optional stability)
                    ord2 = torch.argsort(gid_comp)
                    tok_valid2 = tok_valid.index_select(0, ord2)
                    gid_comp2 = gid_comp.index_select(0, ord2)
                    # Recompute lengths per compact id
                    lens2 = torch.bincount(gid_comp2, minlength=G_eff)
                    Lmax = int(lens2.max().item()) if G_eff > 0 else 0

                    # Cap portfolio length if configured
                    if (self.max_portfolio_len is not None) and (Lmax > self.max_portfolio_len):
                        Lmax = int(self.max_portfolio_len)

                    # Pad into [G_eff, Lmax, Hb]
                    Hb = tok_valid2.size(1)
                    pad_tok = torch.zeros((G_eff, Lmax, Hb), device=device, dtype=dtype)
                    pad_mask = torch.ones((G_eff, Lmax), device=device, dtype=torch.bool)  # True=pad
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

                    # Early exit: if all L<=1, skip fusion entirely to behave like identity (singleton stability)
                    all_singletons = bool((lens2 <= 1).all().item()) if lens2.numel() > 0 else True
                    if all_singletons:
                        # Do not change z_anchor; no portfolio fusion applied
                        tok_ctx_full = None
                    else:
                        tok_ctx = self.portfolio_encoder(pad_tok, src_key_padding_mask=pad_mask)
                        # Unpad back to flat valid order
                        ctx_list = []
                        for g_idx in range(G_eff):
                            Lg = int(lens2[g_idx].item())
                            if Lg == 0:
                                continue
                            Luse = min(Lg, Lmax)
                            ctx_list.append(tok_ctx[g_idx, :Luse, :])
                        tok_ctx_valid = torch.cat(ctx_list, dim=0) if ctx_list else tok_valid2

                        # Restore to original B order
                        # First invert ord2 within valid, then scatter back into B with zeros for invalid
                        inv_ord2 = torch.empty_like(ord2)
                        inv_ord2[ord2] = torch.arange(ord2.numel(), device=device)
                        tok_ctx_valid_back = tok_ctx_valid.index_select(0, inv_ord2)
                        tok_ctx_sorted = torch.zeros_like(tok_sorted)
                        tok_ctx_sorted.index_copy_(0, valid_idx, tok_ctx_valid_back)
                        inv_order = torch.empty_like(order)
                        inv_order[order] = torch.arange(B, device=device)
                        tok_ctx_full = tok_ctx_sorted.index_select(0, inv_order)

                    # Fuse to hidden dim and gate (only if context was computed)
                    if tok_ctx_full is not None:
                        z_ctx_h = self.portfolio_fuse(tok_ctx_full)
                        gamma = torch.sigmoid(self.portfolio_gate) if self.portfolio_gate is not None else 1.0
                        if self.portfolio_attn_mode == "residual":
                            z_anchor = z_anchor + gamma * z_ctx_h
                        else:  # concat
                            z_portfolio_ctx = gamma * z_ctx_h
                # else: no valid gids; skip
        return z_anchor, z_portfolio_ctx
