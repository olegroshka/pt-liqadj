from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
from typer.testing import CliRunner

from ptliq.data.simulate import simulate, SimParams
from ptliq.data.split import compute_default_ranges, write_ranges

# GNN CLIs
from ptliq.cli.gnn_build_graph import app as gnn_build_graph_app
from ptliq.cli.gnn_build_dataset import app as gnn_build_dataset_app
from ptliq.cli.gnn_train import app as gnn_train_app


runner = CliRunner(mix_stderr=False)


def test_gnn_pipeline_e2e(tmp_path: Path):
    # 1) Simulate tiny dataset
    sim = simulate(SimParams(n_bonds=120, n_days=4, providers=["P1"], seed=777, outdir=tmp_path))
    bonds_pq = tmp_path / "bonds.parquet"
    trades_pq = tmp_path / "trades.parquet"
    sim["bonds"].to_parquet(bonds_pq, index=False)
    sim["trades"].to_parquet(trades_pq, index=False)
    assert bonds_pq.exists() and trades_pq.exists()

    # 2) Time-based splits (write_ranges returns a FILE path)
    ranges_path = write_ranges(
        compute_default_ranges(trades_pq, train_end="2025-01-03", val_end="2025-01-04"),
        tmp_path / "splits",
    )
    assert ranges_path.exists() and ranges_path.name in ("ranges.json", "ranges.yaml")
    splits_dir = ranges_path.parent  # downstream steps expect a directory

    # 3) Build graph
    graph_dir = tmp_path / "graph"
    res = runner.invoke(
        gnn_build_graph_app,
        ["--bonds", str(bonds_pq), "--outdir", str(graph_dir)],
    )
    assert res.exit_code == 0, res.stdout or res.stderr

    # Required graph artifacts (current builder always writes these)
    required = (
        "issuer_groups.pt",
        "sector_groups.pt",
        "node_to_issuer.pt",
        "node_to_sector.pt",
    )
    for fname in required:
        assert (graph_dir / fname).exists(), f"missing graph artifact: {fname}"

    # Meta file is optional; accept any common name if present
    maybe_meta = [
        "meta.json",
        "graph_meta.json",
        "meta.yaml",
        "meta.yml",
        "meta.pkl",
        "meta.pt",
    ]
    meta_present = any((graph_dir / m).exists() for m in maybe_meta)
    # Don't fail the test if meta isn't emitted; the dataset step doesn't require it.

    # 4) Build dataset bundles
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
    for split in ("train", "val", "test"):
        assert (ds_dir / f"{split}.pt").exists(), f"missing {split}.pt"

    # 5) Train GNN (small CPU config)
    cfg = {
        "project": {"run_id": "gnn_e2e", "seed": 42},
        "paths": {"dataset_dir": str(ds_dir)},
        "model": {
            "node_id_dim": 16, "nhead": 2, "n_layers": 1, "d_model": 64,
            "gnn_num_hidden": 32, "gnn_out_dim": 64, "gnn_dropout": 0.0,
            "head_hidden": 64, "head_dropout": 0.0,
        },
        "train": {"device": "cpu", "max_epochs": 3, "batch_size": 256, "lr": "1e-3", "patience": 2, "seed": 42},
    }
    cfg_path = tmp_path / "gnn_cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    outdir = tmp_path / "models" / "gnn_e2e"
    res = runner.invoke(
        gnn_train_app,
        ["--data-dir", str(ds_dir), "--config", str(cfg_path), "--outdir", str(outdir)],
    )
    assert res.exit_code == 0, res.stdout or res.stderr

    # 6) Artifacts + metrics checks
    ckpt = outdir / "ckpt.pt"
    ckpt_gnn = outdir / "ckpt_gnn.pt"
    met = outdir / "metrics_val.json"

    assert ckpt.exists(), "ckpt.pt not written"
    assert ckpt_gnn.exists(), "ckpt_gnn.pt not written"
    assert met.exists(), "metrics_val.json not written"

    js = json.loads(met.read_text())
    assert "best_epoch" in js and "history" in js, f"bad metrics keys: {js}"
    mae = js.get("best_val_mae_bps", js.get("val_mae", js.get("val_mae_bps", np.inf)))
    assert np.isfinite(mae), f"non-finite val MAE: {mae}"
    hist = js["history"].get("val_mae_bps", [])
    assert isinstance(hist, list) and len(hist) >= 1
    assert all(np.isfinite(h) for h in hist), f"non-finite entries in history: {hist}"
