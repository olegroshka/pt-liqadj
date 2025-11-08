from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from fastapi.testclient import TestClient

from ptliq.training.gru_loop import train_gru, GRUTrainConfig, GRUModelConfig
from ptliq.cli.train_baseline import app_main as mlp_train_main
from ptliq.service.scoring import GRUScorer, MLPScorer
from ptliq.service.app import create_app


def _make_gru_features(tmp: Path, T: int = 20, F: int = 3) -> Path:
    base = tmp / "features_gru_api"
    base.mkdir(parents=True, exist_ok=True)
    rs = np.random.default_rng(5)
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
            y.append(float(6.0 * float(mv @ W) + 1.5 * float(side[i]) + 0.7 * float(lsz[i]) + rs.normal(0, 0.05)))
        return pd.DataFrame({
            "trade_date": dates[di].astype("datetime64[ns]"),
            "isin": ["SIM" + str(i % 5) for i in range(n)],
            "side_sign": side.astype(float),
            "log_size": lsz.astype(float),
            "y_bps": np.array(y, dtype=float),
        })

    (base / "train.parquet").write_bytes(_mk(128).to_parquet(index=False))
    (base / "val.parquet").write_bytes(_mk(32).to_parquet(index=False))
    (base / "test.parquet").write_bytes(_mk(32).to_parquet(index=False))
    return base


def _make_mlp_features(tmp: Path, run_id: str = "exp_api") -> Path:
    base = tmp / "features" / run_id
    base.mkdir(parents=True, exist_ok=True)
    rs = np.random.default_rng(2)
    n_train, n_val = 96, 32
    f1_tr = rs.normal(0.0, 1.0, size=n_train)
    f2_tr = rs.normal(0.0, 1.0, size=n_train)
    y_tr = 1.5 * f1_tr - 2.5 * f2_tr + rs.normal(0.0, 0.1, size=n_train)
    f1_va = rs.normal(0.0, 1.0, size=n_val)
    f2_va = rs.normal(0.0, 1.0, size=n_val)
    y_va = 1.5 * f1_va - 2.5 * f2_va + rs.normal(0.0, 0.1, size=n_val)

    df_tr = pd.DataFrame({"f_one": f1_tr.astype(float), "f_two": f2_tr.astype(float), "y_bps": y_tr.astype(float)})
    df_va = pd.DataFrame({"f_one": f1_va.astype(float), "f_two": f2_va.astype(float), "y_bps": y_va.astype(float)})
    (base / "train.parquet").write_bytes(df_tr.to_parquet(index=False))
    (base / "val.parquet").write_bytes(df_va.to_parquet(index=False))
    return base


def test_api_returns_same_scale_as_scorer_gru(tmp_path: Path):
    feature_dir = _make_gru_features(tmp_path)
    out = tmp_path / "gru_out_api"
    cfg = GRUTrainConfig(
        feature_dir=str(feature_dir),
        outdir=str(out),
        device="cpu",
        epochs=2,
        lr=5e-3,
        batch_size=64,
        patience=0,
        seed=7,
        early_stopping=False,
        model=GRUModelConfig(hidden=32, layers=1, dropout=0.0, window=2, trade_cols=["side_sign","log_size"]),
    )
    train_gru(cfg)

    scorer = GRUScorer.from_dir(out, device="cpu")
    app = create_app(scorer)
    client = TestClient(app)

    rows = [
        {"isin": "SIMX", "side": "buy", "size": 100000, "asof_date": None},
        {"isin": "SIMY", "side": "sell", "size": 25000,  "asof_date": None},
    ]
    y_local = scorer.score_many(rows).astype(float)

    resp = client.post("/score", json={"rows": rows})
    assert resp.status_code == 200
    data = resp.json()
    preds = [float(p["pred_bps"]) for p in data["preds_bps"]]

    assert len(preds) == len(rows)
    # Parity API vs scorer
    assert np.allclose(preds, y_local, rtol=1e-6, atol=1e-6)
    # Sanity: bps magnitude (not 0.003-style percent)
    assert np.all(np.abs(preds) < 1000.0)


def test_api_returns_same_scale_as_scorer_mlp(tmp_path: Path):
    run_id = "exp_api"
    features_base = _make_mlp_features(tmp_path, run_id=run_id)
    models_dir = tmp_path / "models"
    mlp_train_main(
        features_dir=features_base.parent,
        run_id=run_id,
        models_dir=models_dir,
        device="cpu",
        max_epochs=2,
        batch_size=64,
        lr=1e-3,
        patience=2,
        seed=5,
        hidden="16,8",
        dropout=0.0,
        config=None,
        workdir=tmp_path,
        verbose=False,
    )
    outdir = models_dir / run_id

    scorer = MLPScorer.from_dir(outdir, device="cpu")
    app = create_app(scorer)
    client = TestClient(app)

    rows = [
        {"f_one": 0.25, "f_two": -0.75},
        {"f_one": -0.5, "f_two": 0.3},
    ]
    y_local = scorer.score_many(rows).astype(float)

    resp = client.post("/score", json={"rows": rows})
    assert resp.status_code == 200
    data = resp.json()
    preds = [float(p["pred_bps"]) for p in data["preds_bps"]]

    assert len(preds) == len(rows)
    assert np.allclose(preds, y_local, rtol=1e-6, atol=1e-6)
    assert np.all(np.abs(preds) < 1000.0)
