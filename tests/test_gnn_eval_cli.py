# ptliq/cli/gnn_eval.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union, Optional
import json
import typer
import torch
import torch.nn.functional as F

# local lightweight helpers (kept inline to avoid importing training loop)
def _nums_first_tensor(nums: Union[torch.Tensor, Dict[str, torch.Tensor], None]) -> torch.Tensor:
    if nums is None:
        raise RuntimeError("nums is None; cannot infer dimensionality")
    if torch.is_tensor(nums):
        return nums
    for v in nums.values():
        if torch.is_tensor(v):
            return v
    raise RuntimeError("nums dict had no tensors")

def _index_any(obj: Any, sl: slice) -> Any:
    if obj is None:
        return None
    if torch.is_tensor(obj):
        return obj[sl]
    if isinstance(obj, dict):
        return {k: v[sl] for k, v in obj.items()}
    return obj

def _pack_slice(gi: Any, sl: slice) -> Dict[str, Any]:
    batch: Dict[str, Any] = {
        "node_ids": gi.node_ids[sl],
        "cats": {k: v[sl] for k, v in gi.cats.items()},
        "nums": _index_any(gi.nums, sl),
        "issuer_groups": gi.issuer_groups,
        "sector_groups": gi.sector_groups,
        "node_to_issuer": gi.node_to_issuer,
        "node_to_sector": gi.node_to_sector,
        "port_nodes": gi.port_nodes[sl] if getattr(gi, "port_nodes", torch.empty(0)).numel() > 0 else gi.port_nodes,
        "port_len": gi.port_len[sl] if getattr(gi, "port_len", torch.empty(0)).numel() > 0 else gi.port_len,
        "y": gi.y[sl],
    }
    ssu = getattr(gi, "size_side_urg", None)
    if ssu is not None:
        batch["size_side_urg"] = ssu[sl]
    return batch

def _to_dev(b: Dict[str, Any], dev: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in b.items():
        if torch.is_tensor(v):
            out[k] = v.to(dev)
        elif isinstance(v, dict):
            out[k] = {kk: vv.to(dev) for kk, vv in v.items()}
        else:
            out[k] = v
    return out

# ---------------- CLI ----------------
app = typer.Typer(no_args_is_help=True)

def _load_bundle(p: Path) -> Any:
    # our trainer saved a rich GraphInputs object; safe to load with map_location=cpu
    return torch.load(p, map_location="cpu")

def _pick_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

@app.command()
def app_main(
    data_dir: Path = typer.Option(..., "--data-dir", help="Folder with {train,val,test}.pt bundles"),
    ckpt: Path = typer.Option(..., "--ckpt", help="Path to ckpt.pt (or ckpt_gnn.pt)"),
    outdir: Path = typer.Option(..., "--outdir", help="Where to write metrics_test.json"),
    config: Optional[Path] = typer.Option(None, "--config", help="(unused) kept for interface symmetry"),
    device: str = typer.Option("auto", help="cpu | cuda | auto"),
    batch_size: int = typer.Option(256, help="Eval batch size"),
):
    """
    Evaluate a saved GNN model checkpoint on test.pt and write metrics_test.json
    """
    data_dir = Path(data_dir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    test_path = data_dir / "test.pt"
    if not test_path.exists():
        raise FileNotFoundError(f"Expected test bundle at {test_path}")

    gi_te = _load_bundle(test_path)

    # Build the model exactly like train_gnn does (cat specs inferred from gi)
    from ptliq.model import PortfolioResidualModel, ModelConfig, NodeFieldSpec  # heavy import

    n_num = int(_nums_first_tensor(gi_te.nums).shape[1])

    cat_specs = []
    if "sector_code" in gi_te.cats:
        vmax = int(gi_te.cats["sector_code"].max().item()) + 1
        cat_specs.append(NodeFieldSpec("sector_code", vmax, 16))
    if "rating_code" in gi_te.cats:
        vmax = int(gi_te.cats["rating_code"].max().item()) + 1
        cat_specs.append(NodeFieldSpec("rating_code", vmax, 12))

    # Try to read training-time config from ckpt to keep dims consistent; otherwise fallback to defaults
    ckpt = torch.load(ckpt, map_location="cpu")
    cfg_in = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    node_id_dim = int(cfg_in.get("node_id_dim", 32))
    gnn_num_hidden = int(cfg_in.get("gnn_num_hidden", 64))
    gnn_out_dim = int(cfg_in.get("gnn_out_dim", 128))
    d_model = int(cfg_in.get("d_model", 128))
    nhead = int(cfg_in.get("nhead", 4))
    n_layers = int(cfg_in.get("n_layers", 1))
    use_baseline = bool(cfg_in.get("use_baseline", False))

    mcfg = ModelConfig(
        n_nodes=int(gi_te.n_nodes),
        node_id_dim=node_id_dim,
        cat_specs=cat_specs,
        n_num=n_num,
        gnn_num_hidden=gnn_num_hidden,
        gnn_out_dim=gnn_out_dim,
        d_model=d_model,
        nhead=nhead,
        n_layers=n_layers,
        device=device,
        use_baseline=use_baseline,
    )
    model = PortfolioResidualModel(mcfg)
    model.load_state_dict(ckpt["state_dict"])
    dev = _pick_device(device)
    model.to(dev).eval()

    # Eval loop
    B = int(gi_te.node_ids.shape[0])
    bs = int(batch_size)
    preds = []
    trues = []
    with torch.no_grad():
        for start in range(0, B, bs):
            end = min(start + bs, B)
            b = _to_dev(_pack_slice(gi_te, slice(start, end)), dev)
            out = model(
                b["node_ids"], b["cats"], b["nums"],
                b["issuer_groups"], b["sector_groups"],
                b["node_to_issuer"], b["node_to_sector"],
                b.get("port_nodes"), b.get("port_len"),
                b.get("size_side_urg"), None,  # baseline_feats=None
            )
            preds.append(out["mean"].view(-1).detach().cpu())
            trues.append(b["y"].view(-1).detach().cpu())

    pred_all = torch.cat(preds) if preds else torch.empty(0)
    true_all = torch.cat(trues) if trues else torch.empty(0)
    mae = float(F.l1_loss(pred_all, true_all).item()) if pred_all.numel() else float("inf")
    metrics = {"n": int(true_all.numel()), "mae_bps": mae}

    (outdir / "metrics_test.json").write_text(json.dumps(metrics, indent=2))
    typer.echo(f"[gnn-eval] Wrote metrics_test.json (n={metrics['n']}, mae_bps={metrics['mae_bps']:.4f}) to {outdir}")

# expose Typer app
app = app

if __name__ == "__main__":
    app()
