from pathlib import Path
from ptliq.data.simulate import SimParams, simulate
from ptliq.data.split import compute_default_ranges, write_ranges
from ptliq.features.build import build_features
from ptliq.training.loop import TrainConfig, train_loop
from ptliq.training.dataset import discover_features, compute_standardizer, apply_standardizer
import pandas as pd
import json
import numpy as np

def test_train_eval_smoke(tmp_path: Path):
    # simulate tiny dataset
    frames = simulate(SimParams(n_bonds=60, n_days=4, providers=["P1"], seed=11, outdir=tmp_path))
    (tmp_path / "bonds.parquet").write_bytes(frames["bonds"].to_parquet(index=False))
    (tmp_path / "trades.parquet").write_bytes(frames["trades"].to_parquet(index=False))

    # splits: 2 days train, 1 day val, rest test
    r = compute_default_ranges(tmp_path / "trades.parquet", "2025-01-03", "2025-01-04")
    rpath = write_ranges(r, tmp_path / "splits")

    # features
    feats = build_features(tmp_path, rpath)
    assert "train" in feats and "val" in feats and "test" in feats

    # standardize and train
    feat_cols = discover_features(feats["train"])
    stdz = compute_standardizer(feats["train"], feat_cols)
    Xtr = apply_standardizer(feats["train"], feat_cols, stdz)
    Xva = apply_standardizer(feats["val"], feat_cols, stdz)
    ytr = feats["train"]["y_bps"].astype(float).to_numpy()
    yva = feats["val"]["y_bps"].astype(float).to_numpy()

    outdir = tmp_path / "model"
    cfg = TrainConfig(max_epochs=2, batch_size=512, lr=1e-3, patience=1, device="cpu", hidden=[16])
    res = train_loop(Xtr, ytr, Xva, yva, feat_cols, outdir, cfg)

    # artifacts exist
    assert (outdir / "ckpt.pt").exists()
    assert (outdir / "metrics_val.json").exists()
    with open(outdir / "metrics_val.json", "r", encoding="utf-8") as f:
        js = json.load(f)
    assert "best_epoch" in js and "history" in js
