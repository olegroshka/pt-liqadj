from pathlib import Path
import json
import numpy as np
from fastapi.testclient import TestClient
from ptliq.service.scoring import MLPScorer
from ptliq.service.app import create_app
from ptliq.data.simulate import SimParams, simulate
from ptliq.data.split import compute_default_ranges, write_ranges
from ptliq.features.build import build_features
from ptliq.training.loop import TrainConfig, train_loop
from ptliq.training.dataset import discover_features, compute_standardizer, apply_standardizer

def test_service_health_and_score(tmp_path: Path):
    # train small model
    frames = simulate(SimParams(n_bonds=40, n_days=3, providers=["P1"], seed=44, outdir=tmp_path))
    frames["bonds"].to_parquet(tmp_path / "bonds.parquet", index=False)
    frames["trades"].to_parquet(tmp_path / "trades.parquet", index=False)

    # train=2025-01-02, val=2025-01-03, test=2025-01-04
    r = compute_default_ranges(tmp_path / "trades.parquet", "2025-01-02", "2025-01-03")
    rpath = write_ranges(r, tmp_path / "splits")
    feats = build_features(tmp_path, rpath)

    feat_cols = discover_features(feats["train"])
    stdz = compute_standardizer(feats["train"], feat_cols)
    Xtr = apply_standardizer(feats["train"], feat_cols, stdz)
    Xva = apply_standardizer(feats["val"], feat_cols, stdz)
    ytr = feats["train"]["y_bps"].to_numpy().astype(np.float32)
    yva = feats["val"]["y_bps"].to_numpy().astype(np.float32)

    mdir = tmp_path / "models/run"
    mdir.parent.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig(max_epochs=2, batch_size=256, lr=1e-3, patience=1, device="cpu", hidden=[16])
    train_loop(Xtr, ytr, Xva, yva, feat_cols, mdir, cfg)
    (mdir / "scaler.json").write_text(json.dumps({"mean": stdz["mean"].tolist(), "std": stdz["std"].tolist()}))
    (mdir / "feature_names.json").write_text(json.dumps(feat_cols))
    (mdir / "train_config.json").write_text(json.dumps({"hidden": [16], "dropout": 0.0}))

    scorer = MLPScorer.from_dir(mdir, device="cpu")
    app = create_app(scorer)
    client = TestClient(app)

    # health
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True

    # score: use first 3 rows from test split, include isin to preserve mapping
    test_df = feats["test"][["isin"] + feat_cols].head(3)
    rows = test_df.to_dict(orient="records")
    r = client.post("/score", json={"rows": rows})
    assert r.status_code == 200
    js = r.json()
    assert "preds_bps" in js and len(js["preds_bps"]) == len(rows)
    # ensure each item has fields and the corresponding isin in the same order
    isins = test_df["isin"].astype(str).tolist()
    got_isins = [d.get("isin") for d in js["preds_bps"]]
    assert got_isins == isins
    # pred_bps is a float number
    assert all(isinstance(d.get("pred_bps"), (int, float)) for d in js["preds_bps"]) 
