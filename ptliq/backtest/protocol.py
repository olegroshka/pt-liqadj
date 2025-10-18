# ptliq/backtest/protocol.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import json
import numpy as np
import torch

from ptliq.training.dataset import load_split_parquet, discover_features, apply_standardizer
from ptliq.training.loop import load_model_for_eval
from .metrics import basic_metrics, slice_by_column, decile_slices, calibration_bins

def run_backtest(features_dir: Path, run_id: str, models_dir: Path, outdir: Path) -> Dict[str, Any]:
    features_dir = Path(features_dir) / run_id
    models_dir = Path(models_dir) / run_id
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    # load data
    test_df = load_split_parquet(features_dir, "test")
    feat_cols = discover_features(test_df)
    if not feat_cols:
        raise RuntimeError("No features found for test set.")

    # load scaler
    with open(models_dir / "scaler.json", "r", encoding="utf-8") as f:
        sc = json.load(f)
    mean = np.array(sc["mean"], dtype=np.float32); std = np.array(sc["std"], dtype=np.float32)
    stdz = {"mean": mean, "std": std}

    X_test = apply_standardizer(test_df, feat_cols, stdz)
    y_test = test_df["y_bps"].astype(float).to_numpy().astype(np.float32)

    # try to read hidden/dropout from saved train_config
    hidden = [64, 64]
    dropout = 0.0
    cfg_path = models_dir / "train_config.json"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            tr_cfg = json.load(f)
        hidden = tr_cfg.get("hidden", hidden)
        dropout = tr_cfg.get("dropout", dropout)

    # load model with correct architecture
    model, dev = load_model_for_eval(models_dir, in_dim=X_test.shape[1], hidden=hidden, dropout=dropout, device="cpu")
    with torch.no_grad():
        yhat = model(torch.from_numpy(X_test).to(dev)).cpu().numpy().astype(np.float32)

    # residuals frame
    res = test_df[["ts", "isin"]].copy()
    res["y_bps"] = y_test
    res["yhat_bps"] = yhat
    # include a couple of useful features for slicing
    for c in ["f_sector_code", "f_rating_code", "f_size_log"]:
        if c in test_df.columns:
            res[c] = test_df[c].values

    # metrics
    overall = basic_metrics(y_test, yhat)
    by_sector = slice_by_column(res.dropna(subset=["f_sector_code"]), "f_sector_code") if "f_sector_code" in res.columns else {}
    size_deciles = decile_slices(res.dropna(subset=["f_size_log"]), "f_size_log") if "f_size_log" in res.columns else {}
    calib = calibration_bins(y_test, yhat, n_bins=10).to_dict(orient="list")

    metrics = {
        "overall": overall,
        "by_sector_code": by_sector,
        "by_size_decile": size_deciles,
        "calibration": calib,
        "n": int(len(res)),
        "features_used": feat_cols,
    }

    # write artifacts
    out_parquet = outdir / "residuals.parquet"
    res.to_parquet(out_parquet, index=False)
    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics
