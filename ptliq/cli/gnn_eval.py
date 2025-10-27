# ptliq/cli/gnn_eval.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

import json
import typer
import torch

from ptliq.training.gnn_loop import GraphInputs, _nums_first_tensor, _pick_device
from ptliq.model import PortfolioResidualModel, ModelConfig, NodeFieldSpec

typer_app = typer.Typer(no_args_is_help=True)

def _load_bundle(p: Path) -> GraphInputs:
    return torch.load(p, map_location="cpu", weights_only=False)

def _rebuild_model_from_ckpt_and_data(ckpt: Dict[str, Any], gi: GraphInputs, device: str) -> PortfolioResidualModel:
    cfg_dict = ckpt.get("config", {})
    n_num = int(_nums_first_tensor(gi.nums).shape[1]) if gi.nums is not None else 0

    cat_specs: List[NodeFieldSpec] = []
    if "sector_code" in gi.cats:
        vmax = int(gi.cats["sector_code"].max().item()) + 1
        cat_specs.append(NodeFieldSpec("sector_code", vmax, 16))
    if "rating_code" in gi.cats:
        vmax = int(gi.cats["rating_code"].max().item()) + 1
        cat_specs.append(NodeFieldSpec("rating_code", vmax, 12))

    mcfg = ModelConfig(
        n_nodes=int(gi.n_nodes),
        node_id_dim=int(cfg_dict.get("node_id_dim", 32)),
        cat_specs=cat_specs,
        n_num=n_num,
        gnn_num_hidden=int(cfg_dict.get("gnn_num_hidden", 64)),
        gnn_out_dim=int(cfg_dict.get("gnn_out_dim", 128)),
        d_model=int(cfg_dict.get("d_model", 128)),
        nhead=int(cfg_dict.get("nhead", 4)),
        n_layers=int(cfg_dict.get("n_layers", 1)),
        device=device,
        use_baseline=bool(cfg_dict.get("use_baseline", False)),
    )
    model = PortfolioResidualModel(mcfg)
    model.load_state_dict(ckpt["state_dict"])
    return model

@typer_app.command()
def main(
    data_dir: Path = typer.Option(..., "--data-dir", help="Folder containing {test}.pt"),
    ckpt: Path = typer.Option(..., help="Checkpoint file produced by ptliq-gnn-train (ckpt.pt)"),
    outdir: Path = typer.Option(Path("models/exp_gnn"), help="Where to write eval metrics"),
    device: str = typer.Option("auto", help="'cpu' | 'cuda' | 'auto'"),
    batch_size: int = typer.Option(256, help="Eval batch size"),
):
    """
    Evaluate a trained GNN model on test.pt from --data-dir.
    Writes metrics_test.json.
    """
    data_dir = Path(data_dir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    test_path = data_dir / "test.pt"
    if not test_path.exists():
        raise FileNotFoundError(f"Missing dataset bundle: {test_path}")

    gi: GraphInputs = _load_bundle(test_path)
    dev = _pick_device(device)
    ckpt_obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = _rebuild_model_from_ckpt_and_data(ckpt_obj, gi, device=device).to(dev)
    model.eval()

    B = int(gi.node_ids.shape[0])
    bs = int(batch_size)
    preds = []
    trues = []

    with torch.no_grad():
        for start in range(0, B, bs):
            end = min(start + bs, B)
            sl = slice(start, end)

            node_ids = gi.node_ids[sl].to(dev)
            cats = {k: v[sl].to(dev) for k, v in gi.cats.items()}
            if gi.nums is None:
                nums = None
            elif torch.is_tensor(gi.nums):
                nums = gi.nums[sl].to(dev)
            else:
                nums = {k: v[sl].to(dev) for k, v in gi.nums.items()}

            port_nodes = gi.port_nodes[sl].to(dev) if gi.port_nodes.numel() > 0 else gi.port_nodes
            port_len = gi.port_len[sl].to(dev) if gi.port_len.numel() > 0 else gi.port_len
            size_side_urg = getattr(gi, "size_side_urg", None)
            size_side_urg = size_side_urg[sl].to(dev) if size_side_urg is not None else None

            out = model(
                node_ids, cats, nums,
                gi.issuer_groups, gi.sector_groups,
                gi.node_to_issuer, gi.node_to_sector,
                port_nodes, port_len, size_side_urg, None,
            )
            preds.append(out["mean"].view(-1).detach().cpu())
            trues.append(gi.y[sl].view(-1).detach().cpu())

    pred_all = torch.cat(preds) if preds else torch.empty(0)
    true_all = torch.cat(trues) if trues else torch.empty(0)
    if pred_all.numel():
        diff = pred_all - true_all
        mae = float(torch.mean(torch.abs(diff)).item())
        rmse = float(torch.sqrt(torch.mean(diff * diff)).item())
    else:
        mae = float("inf")
        rmse = float("inf")

    # try to read best metrics if present (support both flat and nested under 'best')
    best_epoch = int(ckpt_obj.get("best_epoch", -1))
    best_val_mae_bps = ckpt_obj.get("best_val_mae_bps")
    best_val_rmse_bps = ckpt_obj.get("best_val_rmse_bps")
    if "best" in ckpt_obj and isinstance(ckpt_obj["best"], dict):
        best_epoch = int(ckpt_obj["best"].get("best_epoch", ckpt_obj["best"].get("epoch", best_epoch)))
        best_val_mae_bps = ckpt_obj["best"].get("best_val_mae_bps", ckpt_obj["best"].get("val_mae", best_val_mae_bps))
        best_val_rmse_bps = ckpt_obj["best"].get("best_val_rmse_bps", best_val_rmse_bps)

    metrics = {
        "n": int(true_all.numel()),
        "mae_bps": mae,
        "rmse_bps": rmse,
        "best_val_mae_bps": float(best_val_mae_bps) if best_val_mae_bps is not None else float("nan"),
        "best_val_rmse_bps": float(best_val_rmse_bps) if best_val_rmse_bps is not None else float("nan"),
        "best_epoch": int(best_epoch),
    }
    (outdir / "metrics_test.json").write_text(json.dumps(metrics, indent=2))
    typer.echo(
        f"[gnn-eval] n={metrics['n']} test_mae_bps={metrics['mae_bps']:.6f} test_rmse_bps={metrics['rmse_bps']:.6f} "
        f"(best_val_mae={metrics['best_val_mae_bps']:.6f}, best_val_rmse={metrics['best_val_rmse_bps']:.6f} @ {metrics['best_epoch']})"
    )

# expose Typer app object (console script calls this)
app = typer_app

if __name__ == "__main__":
    typer_app()
