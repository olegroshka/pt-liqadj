from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

import typer
import torch
import json

from ptliq.model.mv_dgt import MultiViewDGT


app = typer.Typer(no_args_is_help=True)


def _load_meta(workdir: Path) -> dict:
    meta_path = workdir / "mvdgt_meta.json"
    if not meta_path.exists():
        raise typer.BadParameter(f"mvdgt_meta.json not found in workdir: {workdir}")
    try:
        return json.loads(meta_path.read_text())
    except Exception as e:
        raise typer.BadParameter(f"Failed to read {meta_path}: {e}")


def _load_view_masks(workdir: Path) -> dict[str, torch.Tensor]:
    vm_path = workdir / "view_masks.pt"
    if not vm_path.exists():
        raise typer.BadParameter(f"view_masks.pt not found in workdir: {workdir}")
    # Explicitly allow object tensors by turning off weights_only for compatibility
    vm = torch.load(vm_path, map_location="cpu", weights_only=False)
    if not isinstance(vm, dict):
        raise typer.BadParameter(f"view_masks.pt must be a dict[str,Tensor]; got {type(vm)}")
    # normalize to bool 1D
    out: dict[str, torch.Tensor] = {}
    for k, v in vm.items():
        t = v.view(-1)
        if t.dtype != torch.bool:
            t = (t != 0)
        out[k] = t
    return out


def _load_graph_x_e(meta: dict) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    files = meta.get("files", {}) if isinstance(meta, dict) else {}
    pyg_path = Path(files.get("pyg_graph", ""))
    if not pyg_path.exists():
        raise typer.BadParameter("pyg_graph.pt path missing in mvdgt_meta.json['files']['pyg_graph']")
    data = torch.load(pyg_path, map_location="cpu", weights_only=False)
    x = data.x.float().cpu()
    edge_index = data.edge_index.cpu()
    edge_weight = data.edge_weight.cpu() if hasattr(data, "edge_weight") else None
    return x, edge_index, edge_weight


def _opt_int(val, default: Optional[int] = None) -> Optional[int]:
    """Coerce possibly-null value to Optional[int].
    Returns None when val is None or 0-like and default is None; otherwise uses default.
    """
    if val is None:
        return default
    try:
        iv = int(val)
    except Exception:
        return default
    # Treat 0 as "disabled" only when default is None (used for optional dims)
    if iv == 0 and default is None:
        return None
    return iv


def _safe_int(d: dict, key: str, default: int) -> int:
    v = d.get(key, default)
    if v is None:
        return int(default)
    try:
        return int(v)
    except Exception:
        return int(default)


def _safe_float(d: dict, key: str, default: float) -> float:
    v = d.get(key, default)
    if v is None:
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _resolve_checkpoint(workdir: Path) -> Path | None:
    # New format (saved by training loop)
    ckpt_new = workdir / "ckpt.pt"
    if ckpt_new.exists():
        return ckpt_new
    # Legacy/path used in some runs
    ckpt_legacy = workdir / "mv_dgt_ckpt.pt"
    if ckpt_legacy.exists():
        return ckpt_legacy
    return None


def _load_state_dict(ckpt_path: Path) -> Dict[str, torch.Tensor]:
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        # Common keys: 'state_dict', sometimes nested variants
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return {k: v for k, v in obj["state_dict"].items()}
        # direct state dict
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj  # looks like a raw state_dict
    raise typer.BadParameter(f"Unrecognized checkpoint format in {ckpt_path}")


class _OnnxWrapper(torch.nn.Module):
    """
    Thin wrapper to expose only tensor inputs for ONNX export while delegating
    to MultiViewDGT with fixed optional arguments.
    """

    def __init__(self, base: MultiViewDGT, use_market: bool, mkt_dim: int):
        super().__init__()
        self.base = base
        self.use_market = bool(use_market)
        self.mkt_dim = int(mkt_dim)

    def forward(
        self,
        x: torch.Tensor,
        anchor_idx: torch.Tensor,
        pf_gid: torch.Tensor,
        trade_feat: torch.Tensor,
        market_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mf = market_feat
        if self.use_market and (self.mkt_dim > 0):
            # ensure a tensor is provided for market path
            if mf is None:
                B = anchor_idx.size(0)
                mf = torch.zeros(B, self.mkt_dim, dtype=x.dtype, device=x.device)
        else:
            mf = None
        return self.base(
            x=x,
            anchor_idx=anchor_idx,
            market_feat=mf,
            pf_gid=pf_gid,
            port_ctx=None,
            trade_feat=trade_feat,
            return_aux=False,
        )


@app.callback(invoke_without_command=True)
def app_main(
    workdir: Path = typer.Option(Path("data/mvdgt/dgt_default"), help="MV-DGT working directory with graph/masks/meta/checkpoint"),
    onnx_out: Optional[Path] = typer.Option(None, help="Output .onnx path; defaults to <workdir>/model_architecture.onnx"),
    opset: int = typer.Option(18, help="ONNX opset version for export (>=18 recommended)"),
    batch_size: int = typer.Option(1, help="Dummy batch size for anchor/pf/trade inputs"),
    disable_market: bool = typer.Option(False, help="Disable market branch during export (sets use_market=False)"),
    disable_portfolio_attn: bool = typer.Option(False, help="Disable portfolio self-attention during export"),
):
    """Export a trained MultiViewDGT model to ONNX for Netron visualization.

    Uses the default paper pipeline layout: loads graph/view masks/meta from workdir,
    reconstructs the model, loads weights from ckpt.pt (or mv_dgt_ckpt.pt), and exports
    an ONNX model that you can open in Netron.
    """
    workdir = Path(workdir)
    out_path = Path(onnx_out) if onnx_out is not None else (workdir / "model_architecture.onnx")

    meta = _load_meta(workdir)
    view_masks = _load_view_masks(workdir)
    x, edge_index, edge_weight = _load_graph_x_e(meta)

    # Infer dims and market availability
    files = meta.get("files", {}) if isinstance(meta, dict) else {}
    mkt_dim = 0
    mkt_path_str = files.get("market_context")
    if (not disable_market) and mkt_path_str:
        try:
            mkt_ctx = torch.load(Path(mkt_path_str), map_location="cpu")
            if isinstance(mkt_ctx, dict) and ("mkt_feat" in mkt_ctx):
                mkt_dim = int(mkt_ctx["mkt_feat"].size(1))
        except Exception:
            mkt_dim = 0

    # Try to load saved model_config.json for exact settings; fallback to conservative defaults
    model_cfg_path = workdir / "model_config.json"
    if model_cfg_path.exists():
        try:
            model_cfg = json.loads(model_cfg_path.read_text())
        except Exception:
            model_cfg = {}
    else:
        model_cfg = {}

    hidden = _safe_int(model_cfg, "hidden", 128)
    heads = _safe_int(model_cfg, "heads", 2)
    dropout = _safe_float(model_cfg, "dropout", 0.10)
    trade_dim = _safe_int(model_cfg, "trade_dim", 2)
    use_portfolio = bool(model_cfg.get("use_portfolio", True))
    use_market_flag = model_cfg.get("use_market", None)
    if disable_market:
        use_market = False
        mkt_dim = 0
    else:
        if use_market_flag is None:
            use_market = bool(mkt_dim > 0)
        else:
            use_market = bool(use_market_flag)
    view_names = list(model_cfg.get("views", ["struct", "port", "corr_global", "corr_local"]))

    # Build model on CPU for export
    model = MultiViewDGT(
        x_dim=int(x.size(1)),
        hidden=hidden,
        heads=heads,
        dropout=dropout,
        view_masks=view_masks,
        edge_index=edge_index,
        edge_weight=edge_weight,
        mkt_dim=int(mkt_dim),
        use_portfolio=use_portfolio,
        use_market=use_market,
        trade_dim=trade_dim,
        view_names=view_names,
        use_pf_head=bool(model_cfg.get("use_pf_head", False)),
        pf_head_hidden=model_cfg.get("pf_head_hidden", None),
        use_portfolio_attn=False if disable_portfolio_attn else bool(model_cfg.get("use_portfolio_attn", False)),
        portfolio_attn_layers=int(model_cfg.get("portfolio_attn_layers", 1) or 1),
        portfolio_attn_heads=int(model_cfg.get("portfolio_attn_heads", 4) or 4),
        portfolio_attn_dropout=_safe_float(model_cfg, "portfolio_attn_dropout", float(dropout)) if model_cfg.get("portfolio_attn_dropout", None) is not None else float(dropout),
        portfolio_attn_hidden=_opt_int(model_cfg.get("portfolio_attn_hidden", None), default=None),
        portfolio_attn_concat_trade=bool(model_cfg.get("portfolio_attn_concat_trade", True)),
        portfolio_attn_concat_market=bool(model_cfg.get("portfolio_attn_concat_market", False)),
        portfolio_attn_mode=str(model_cfg.get("portfolio_attn_mode", "residual")),
        portfolio_attn_gate_init=_safe_float(model_cfg, "portfolio_attn_gate_init", 0.0),
        max_portfolio_len=_opt_int(model_cfg.get("max_portfolio_len", None), default=None),
    ).cpu().eval()

    # Load weights (optional but recommended for Netron to show parameterized nodes)
    ckpt_path = _resolve_checkpoint(workdir)
    if ckpt_path is not None:
        try:
            sd = _load_state_dict(ckpt_path)
            model.load_state_dict(sd, strict=False)
        except Exception as e:
            typer.echo(f"[warn] failed to load checkpoint weights from {ckpt_path}: {e}")
    else:
        typer.echo("[warn] checkpoint not found; exporting architecture with random-initialized weights")

    # Build dummy inputs
    B = int(batch_size)
    N, F = int(x.size(0)), int(x.size(1))
    x_dummy = x  # full graph node features
    # A small dummy set of anchors to keep graph reasonably small in Netron view
    anchor_idx = torch.zeros(B, dtype=torch.long)
    pf_gid = torch.full((B,), -1, dtype=torch.long)  # -1 to indicate no portfolio grouping
    trade_feat = torch.zeros(B, trade_dim, dtype=torch.float32)
    market_feat = torch.zeros(B, mkt_dim, dtype=torch.float32) if (use_market and mkt_dim > 0) else None

    wrapper = _OnnxWrapper(model, use_market=use_market, mkt_dim=mkt_dim).cpu().eval()

    # Export
    input_tensors = (x_dummy, anchor_idx, pf_gid, trade_feat)
    input_names = ["x", "anchor_idx", "pf_gid", "trade_feat"]
    dynamic_axes = {
        "anchor_idx": {0: "batch"},
        "pf_gid": {0: "batch"},
        "trade_feat": {0: "batch"},
    }
    if use_market and (mkt_dim > 0):
        input_tensors = (x_dummy, anchor_idx, pf_gid, trade_feat, market_feat)  # type: ignore[assignment]
        input_names = ["x", "anchor_idx", "pf_gid", "trade_feat", "market_feat"]
        dynamic_axes["market_feat"] = {0: "batch"}

    # Use legacy path (dynamo=False) with dynamic_axes; avoid dynamic_shapes for compatibility.
    torch.onnx.export(
        wrapper,
        input_tensors,
        str(out_path),
        export_params=True,
        opset_version=int(opset),
        do_constant_folding=False,
        input_names=input_names,
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )

    typer.echo(f"Model exported to {out_path}")


if __name__ == "__main__":
    app()
