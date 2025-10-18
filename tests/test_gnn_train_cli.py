import json
from pathlib import Path
from typer.testing import CliRunner

from ptliq.data.simulate import simulate, SimParams
from ptliq.data.split import compute_default_ranges, write_ranges
from ptliq.cli.gnn_build_graph import app as gnn_build_graph_app
from ptliq.cli.gnn_build_dataset import app as gnn_build_dataset_app
from ptliq.cli.gnn_train import app as gnn_train_app

runner = CliRunner()

def _mk_splits(trades_pq: Path, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    ranges = compute_default_ranges(trades_pq, train_end="2025-01-03", val_end="2025-01-04")
    return write_ranges(ranges, outdir)

def test_gnn_train_cli_end_to_end(tmp_path: Path):
    # 1) simulate tiny dataset (deterministic)
    frames = simulate(SimParams(n_bonds=120, n_days=4, providers=["P1"], seed=123, outdir=tmp_path))
    bonds_pq = tmp_path / "bonds.parquet"
    trades_pq = tmp_path / "trades.parquet"
    frames["bonds"].to_parquet(bonds_pq, index=False)
    frames["trades"].to_parquet(trades_pq, index=False)

    # 2) splits (file path returned)
    ranges_file = _mk_splits(trades_pq, tmp_path / "splits")
    splits_dir = ranges_file.parent  # dataset CLI expects a directory

    # 3) graph
    graph_dir = tmp_path / "graph"
    res = runner.invoke(gnn_build_graph_app, ["--bonds", str(bonds_pq), "--outdir", str(graph_dir)])
    assert res.exit_code == 0, res.stdout or res.stderr

    # 4) dataset bundles
    ds_dir = tmp_path / "gnn_ds"
    res = runner.invoke(
        gnn_build_dataset_app,
        [
            "--trades", str(trades_pq),
            "--bonds", str(bonds_pq),
            "--splits-dir", str(splits_dir),
            "--graph-dir", str(graph_dir),
            "--outdir", str(ds_dir),
            "--derive-target",
            "--ref-price-col", "clean_price",
            "--default-par", "100",
        ],
    )
    assert res.exit_code == 0, res.stdout or res.stderr
    for split in ("train.pt", "val.pt", "test.pt"):
        assert (ds_dir / split).exists()

    # 5) train
    outdir = tmp_path / "models" / "exp_gnn_cli"
    res = runner.invoke(
        gnn_train_app,
        ["--data-dir", str(ds_dir), "--config", "configs/gnn.default.yaml", "--outdir", str(outdir)],
    )
    assert res.exit_code == 0, res.stdout or res.stderr

    # 6) artifacts + metrics sanity
    assert (outdir / "ckpt.pt").exists()
    assert (outdir / "ckpt_gnn.pt").exists()
    mpath = outdir / "metrics_val.json"
    assert mpath.exists()
    metrics = json.loads(mpath.read_text())

    # must at least evaluate once and get a finite MAE
    assert "best_val_mae_bps" in metrics and float(metrics["best_val_mae_bps"]) < float("inf")
    hist = metrics.get("history", {}).get("val_mae_bps", [])
    assert isinstance(hist, list) and len(hist) >= 1
