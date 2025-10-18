from __future__ import annotations
from pathlib import Path
import json, typer, torch, time
from ptliq.cli.gnn_eval import _default_factory, _unwrap_state
from ptliq.training.gnn_loop import GNNTrainConfig
from ptliq.model.utils import GraphInputs
from ptliq.model import resolve_device

app = typer.Typer(no_args_is_help=True)

@app.command()
def app(
    data_dir: Path = typer.Option(..., help="Directory with test.pt (and maybe val.pt)"),
    ckpt: Path = typer.Option(..., help="ckpt_gnn.pt"),
    reports_dir: Path = typer.Option(Path("reports")),
    run_id: str = typer.Option("exp_gnn"),
    device: str = typer.Option("cpu"),
):
    test_path = data_dir / "test.pt"
    gi: GraphInputs = torch.load(test_path, map_location="cpu")

    cfg = GNNTrainConfig(device=device)
    model = _default_factory(gi, cfg)
    state = _unwrap_state(torch.load(ckpt, map_location="cpu"))
    model.load_state_dict(state, strict=True).eval()
    dev = resolve_device(device)
    model.to(dev)

    # simple backtest: predict all, compute MAE, save predictions
    n = int(gi.node_ids.shape[0]); bs = 512
    preds = []
    with torch.no_grad():
        for i in range(0, n, bs):
            sl = slice(i, min(i+bs, n))
            batch = {
                "node_ids": gi.node_ids[sl].to(dev),
                "cats": {k:v[sl].to(dev) for k,v in gi.cats.items()},
                "nums": gi.nums[sl].to(dev) if gi.nums is not None else None,
                "issuer_groups": gi.issuer_groups,
                "sector_groups": gi.sector_groups,
                "node_to_issuer": gi.node_to_issuer,
                "node_to_sector": gi.node_to_sector,
                "port_nodes": gi.port_nodes[sl].to(dev),
                "port_len": gi.port_len[sl].to(dev),
                "size_side_urg": gi.size_side_urg[sl].to(dev) if getattr(gi,"size_side_urg",None) is not None else None,
            }
            outd = model(**batch, baseline_feats=None)
            preds.append(outd["mean"].detach().cpu().squeeze(-1))
    pred = torch.cat(preds,0)
    y = gi.y.cpu().squeeze(-1)

    mae = float(torch.mean(torch.abs(pred - y)))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    outdir = reports_dir / run_id / "backtest" / stamp
    outdir.mkdir(parents=True, exist_ok=True)

    # artifacts
    (outdir / "metrics.json").write_text(json.dumps({"mae_bps": mae, "n": int(y.numel())}, indent=2))
    try:
        import pandas as pd
        df = pd.DataFrame({"pred": pred.numpy(), "y": y.numpy()})
        df.to_parquet(outdir / "predictions.parquet", index=False)
    except Exception:
        pass
    typer.echo(f"Backtest done. MAE={mae:.4f}, wrote {outdir}")

app = app