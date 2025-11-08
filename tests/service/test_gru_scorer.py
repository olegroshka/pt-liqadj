from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch

from ptliq.training.gru_loop import train_gru, GRUTrainConfig, GRUModelConfig
from ptliq.service.scoring import GRUScorer


def _make_baseline_features(tmp: Path, T: int = 24, F: int = 3) -> Path:
    base = tmp / "features_gru"
    base.mkdir(parents=True, exist_ok=True)
    rs = np.random.default_rng(7)
    dates = pd.date_range("2025-01-01", periods=T, freq="D").normalize()
    mkt = pd.DataFrame({
        "asof_date": dates,
        **{f"m{j}": rs.normal(0.0, 1.0, size=T).astype(float) for j in range(F)}
    })
    mkt.to_parquet(base / "market_features.parquet", index=False)

    def _mk(n: int) -> pd.DataFrame:
        di = rs.integers(0, T, size=n)
        side = rs.choice([-1.0, 1.0], size=n)
        lsz = rs.normal(0.0, 1.0, size=n)
        W = rs.normal(0.0, 0.25, size=F)
        y = []
        for i in range(n):
            mv = mkt[[f"m{j}" for j in range(F)]].iloc[int(di[i])].to_numpy()
            y.append(float(6.0 * float(mv @ W) + 2.0 * side[i] + 0.5 * lsz[i] + rs.normal(0, 0.05)))
        return pd.DataFrame({
            "trade_date": dates[di].astype("datetime64[ns]"),
            "isin": ["SIM" + str(i % 5) for i in range(n)],
            "side_sign": side.astype(float),
            "log_size": lsz.astype(float),
            "y_bps": np.array(y, dtype=float),
        })

    (base / "train.parquet").write_bytes(_mk(192).to_parquet(index=False))
    (base / "val.parquet").write_bytes(_mk(64).to_parquet(index=False))
    (base / "test.parquet").write_bytes(_mk(64).to_parquet(index=False))
    return base


def test_gru_scorer_integration(tmp_path: Path):
    # 1) prepare baseline feature_dir
    feature_dir = _make_baseline_features(tmp_path)
    out = tmp_path / "gru_out"

    # 2) train tiny GRU to produce serving artifacts (baseline)
    cfg = GRUTrainConfig(
        feature_dir=str(feature_dir),
        outdir=str(out),
        device="cpu",
        epochs=2,
        lr=5e-3,
        batch_size=64,
        patience=0,
        seed=13,
        early_stopping=False,
        model=GRUModelConfig(hidden=32, layers=1, dropout=0.0, window=2, trade_cols=["side_sign","log_size"]),
    )
    train_gru(cfg)

    # 3) instantiate scorer from training directory
    scorer = GRUScorer.from_dir(out, device="cpu")

    # 4) build a few request rows (MV-DGT style keys); include edge cases
    rows = [
        {"side": "B", "size": 100000.0, "asof_date": "2025-01-05"},
        {"side": "S", "size": 25000.0,  "asof_date": "2025-01-06"},
        {"side": 1,    "size": 0.0,      "asof_date": None},            # no date → fallback
        {"side": -1,   "size": 5.0,      "asof_date": "not-a-date"},  # bad date string
        {"side": None, "size": None,     "asof_date": "2024-12-31"},   # missing features
    ]

    y = scorer.score_many(rows)

    # 5) assertions: shape, dtype, finiteness
    assert isinstance(y, np.ndarray)
    assert y.shape == (len(rows),)
    assert np.all(np.isfinite(y)), f"Non-finite outputs from GRUScorer: {y}"

    # Basic sanity: different inputs should generally not all be identical
    # (weak check; allow small numerical ties)
    assert len(set(map(lambda v: float(f"{v:.6f}"), y.tolist()))) >= 2, (
        "GRUScorer returned identical scores for diverse inputs; check wiring"
    )
