# tests/test_backtest_protocol.py
from pathlib import Path
import json
import numpy as np

from ptliq.data.simulate import SimParams, simulate
from ptliq.data.split import compute_default_ranges, write_ranges
from ptliq.features.build import build_features
from ptliq.training.loop import TrainConfig, train_loop
from ptliq.training.dataset import (
    discover_features,
    compute_standardizer,
    apply_standardizer,
)
from ptliq.backtest.protocol import run_backtest


def test_backtest_end_to_end(tmp_path: Path):
    # 1) simulate tiny dataset (deterministic)
    frames = simulate(
        SimParams(n_bonds=60, n_days=4, providers=["P1"], seed=12, outdir=tmp_path)
    )
    (tmp_path / "bonds.parquet").write_bytes(frames["bonds"].to_parquet(index=False))
    (tmp_path / "trades.parquet").write_bytes(frames["trades"].to_parquet(index=False))

    # 2) time splits: train=first 2 days, val=next 1 day, test=rest (02..05)
    ranges = compute_default_ranges(
        tmp_path / "trades.parquet", train_end="2025-01-03", val_end="2025-01-04"
    )
    rpath = write_ranges(ranges, tmp_path / "splits")

    # 3) build features (in memory)
    feats = build_features(tmp_path, rpath)
    assert {"train", "val", "test"} <= feats.keys()
    feat_cols = discover_features(feats["train"])
    assert len(feat_cols) > 0

    # 4) WRITE features to disk in the layout run_backtest expects:
    #    <features_dir>/<run_id>/{train,val,test}.parquet
    run_id = "run"
    features_dir = tmp_path / run_id
    features_dir.mkdir(parents=True, exist_ok=True)
    feats["train"].to_parquet(features_dir / "train.parquet", index=False)
    feats["val"].to_parquet(features_dir / "val.parquet", index=False)
    feats["test"].to_parquet(features_dir / "test.parquet", index=False)

    # 5) standardize + train a tiny MLP; save under models/<run_id>/
    stdz = compute_standardizer(feats["train"], feat_cols)
    Xtr = apply_standardizer(feats["train"], feat_cols, stdz)
    Xva = apply_standardizer(feats["val"], feat_cols, stdz)
    ytr = feats["train"]["y_bps"].to_numpy().astype(np.float32)
    yva = feats["val"]["y_bps"].to_numpy().astype(np.float32)

    models_dir = tmp_path / "models" / run_id
    models_dir.parent.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig(
        max_epochs=2, batch_size=256, lr=1e-3, patience=1, device="cpu", hidden=[16], dropout=0.0
    )
    train_loop(Xtr, ytr, Xva, yva, feat_cols, models_dir, cfg)

    # write scaler.json and train_config.json (since we bypassed the CLI)
    with open(models_dir / "scaler.json", "w", encoding="utf-8") as f:
        json.dump({"mean": stdz["mean"].tolist(), "std": stdz["std"].tolist()}, f, indent=2)
    with open(models_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump({"hidden": [16], "dropout": 0.0}, f, indent=2)

    # 6) backtest
    reports_dir = tmp_path / "reports" / run_id / "backtest" / "ts"
    metrics = run_backtest(tmp_path, run_id, tmp_path / "models", reports_dir)

    # 7) assertions
    assert (reports_dir / "metrics.json").exists()
    assert (reports_dir / "residuals.parquet").exists()
    with open(reports_dir / "metrics.json", "r", encoding="utf-8") as f:
        js = json.load(f)
    assert "overall" in js and js["overall"]["n"] > 0
    assert {"mae_bps", "rmse_bps", "bias_bps"} <= js["overall"].keys()
