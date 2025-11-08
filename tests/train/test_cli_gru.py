from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch

from ptliq.training.gru_loop import train_gru, GRUTrainConfig, GRUModelConfig


def _make_baseline_features(tmp: Path, T: int = 30, F: int = 4) -> Path:
    base = tmp / "features_gru"
    base.mkdir(parents=True, exist_ok=True)
    # Build market_features.parquet with daily dates
    dates = pd.date_range("2025-01-01", periods=T, freq="D").normalize()
    rng = np.random.default_rng(123)
    mkt = pd.DataFrame({
        "asof_date": dates,
        **{f"m{j}": rng.normal(0.0, 1.0, size=T).astype(float) for j in range(F)}
    })
    mkt.to_parquet(base / "market_features.parquet", index=False)

    # Build train/val/test splits with columns required by trainer
    # Use only two trade features matching scorer expectations: side_sign and log_size
    n_train, n_val, n_test = 300, 80, 80
    def _mk_split(n: int) -> pd.DataFrame:
        di = rng.integers(0, T, size=n)
        side = rng.choice([-1.0, 1.0], size=n)
        lsz = rng.normal(0.0, 1.0, size=n)
        # target depends on market row and trade features
        W = rng.normal(0.0, 0.25, size=F)
        y = []
        for i in range(n):
            t = di[i]
            mv = mkt[[f"m{j}" for j in range(F)]].iloc[t].to_numpy()
            y.append(float(5.0 * float(mv @ W) + 2.0 * side[i] + 0.7 * lsz[i] + rng.normal(0, 0.1)))
        df = pd.DataFrame({
            "trade_date": dates[di].astype("datetime64[ns]"),
            "isin": ["SIM" + str(i % 5) for i in range(n)],
            "side_sign": side.astype(float),
            "log_size": lsz.astype(float),
            "y_bps": np.array(y, dtype=float),
        })
        return df

    df_tr = _mk_split(n_train)
    df_va = _mk_split(n_val)
    df_te = _mk_split(n_test)

    df_tr.to_parquet(base / "train.parquet", index=False)
    df_va.to_parquet(base / "val.parquet", index=False)
    df_te.to_parquet(base / "test.parquet", index=False)
    return base


def test_cli_gru_training(tmp_path: Path):
    feature_dir = _make_baseline_features(tmp_path)
    out = tmp_path / "gru_model"

    # run training for several epochs to both produce artifacts and allow learning
    cfg = GRUTrainConfig(
        feature_dir=str(feature_dir),
        outdir=str(out),
        device="cpu",
        epochs=5,
        lr=1e-2,
        batch_size=64,
        patience=0,
        seed=7,
        early_stopping=False,
        model=GRUModelConfig(hidden=32, layers=1, dropout=0.0, window=2, trade_cols=["side_sign","log_size"]),
    )
    train_gru(cfg)

    # artifacts exist (baseline layout)
    assert (out / "ckpt.pt").exists(), "GRU training should produce a checkpoint"
    assert (out / "metrics_val.json").exists(), "GRU training should write validation metrics"
    assert (out / "config.json").exists(), "Training+model config should be persisted"
    assert (out / "feature_names.json").exists(), "Feature names should be saved"
    assert (out / "scaler_trade.json").exists(), "Trade scaler should be saved"
    assert (out / "scaler_market.json").exists(), "Market scaler should be saved"
    assert (out / "predictions_val.parquet").exists(), "Validation predictions should be saved"

    # verify metrics schema and reasonable values
    metrics = json.loads((out / "metrics_val.json").read_text())
    assert "best_epoch" in metrics and metrics["best_epoch"] >= 1
    assert "best_val_mae_bps" in metrics and np.isfinite(metrics["best_val_mae_bps"]) and metrics["best_val_mae_bps"] < 1e4
