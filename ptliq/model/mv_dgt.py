# mv_dgt.py
from __future__ import annotations
from typing import Dict, Optional
import torch
from torch import nn
try:
    from torch_geometric.nn import TransformerConv
except Exception as e:  # pragma: no cover - optional dependency guard for import-time
    TransformerConv = None  # type: ignore


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
    ):
        super().__init__()
        if TransformerConv is None:
            raise ImportError("torch-geometric is required for MultiViewDGT; install torch-geometric")
        assert view_masks is not None and edge_index is not None, "view_masks and edge_index are required"

        self.use_portfolio = use_portfolio
        self.use_market = use_market and (mkt_dim > 0)
        self.view_names = ["struct", "port", "corr_global", "corr_local"]

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

        # regression head (anchor [+ optional portfolio, market, trade])
        in_head = hidden
        if self.use_portfolio:
            in_head += hidden
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

    def _run_layer(self, x: torch.Tensor, layer: int) -> torch.Tensor:
        # structural baseline
        ei_s = self.edge_index_all[:, getattr(self, "mask_struct")]
        ew_s = self.edge_weight_all[getattr(self, "mask_struct")]

        convs = self.conv1 if layer == 1 else self.conv2
        glog  = self.g1_logit if layer == 1 else self.g2_logit
        g     = {k: torch.sigmoid(v) for k, v in glog.items()}

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
        self, H: torch.Tensor,
        pf_gid: torch.Tensor,
        port_ctx: Optional[dict] = None
    ) -> torch.Tensor:
        """Return [B, hidden] weighted portfolio vectors for each sample (or zeros if pf_gid<0).
        Expects port_ctx with keys: port_nodes_flat, port_w_signed_flat, port_offsets, port_len.
        """
        if (not self.use_portfolio) or (port_ctx is None):
            return torch.zeros((pf_gid.size(0), H.size(1)), device=H.device)

        nodes_flat  = port_ctx["port_nodes_flat"]      # [T]
        w_sgn_flat  = port_ctx["port_w_signed_flat"]   # [T]
        offsets     = port_ctx["port_offsets"]         # [G]
        lengths     = port_ctx["port_len"]             # [G]

        out = torch.zeros((pf_gid.size(0), H.size(1)), device=H.device)
        for i, gid in enumerate(pf_gid.tolist()):
            if gid < 0 or gid >= int(len(lengths)):
                continue
            o = int(offsets[gid].item()); L = int(lengths[gid].item())
            if L == 0:
                continue
            idx = nodes_flat[o:o+L].long()
            w   = w_sgn_flat[o:o+L].view(-1, 1).to(H.device)
            out[i] = (w * H.index_select(0, idx.to(H.device))).sum(dim=0)
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

        # assemble sample-wise heads
        z_anchor = h2.index_select(0, anchor_idx.long())
        z_list = [z_anchor]

        if self.use_portfolio:
            pf = torch.zeros_like(anchor_idx) if pf_gid is None else pf_gid
            z_port = self._portfolio_vectors(h2, pf.long(), port_ctx)
            z_list.append(z_port)

        if self.use_market and (market_feat is not None):
            z_mkt = self.mkt_enc(market_feat)
            z_list.append(z_mkt)

        if (self.trade_enc is not None) and (trade_feat is not None):
            z_trade = self.trade_enc(trade_feat)
            z_list.append(z_trade)

        z = torch.cat(z_list, dim=1)
        yhat = self.head(z).squeeze(-1)   # [B]
        return yhat
