from pathlib import Path
import json
import pandas as pd
import numpy as np
from ptliq.data.simulate import SimParams, simulate
from ptliq.data.split import compute_default_ranges, write_ranges
from ptliq.features.build import build_features
from ptliq.training.loop import TrainConfig, train_loop
from ptliq.training.dataset import discover_features, compute_standardizer, apply_standardizer
from ptliq.cli.pack import app as pack_app
from ptliq.cli.score import app as score_app
from typer.testing import CliRunner

runner = CliRunner(mix_stderr=False)

def test_pack_and_score_offline(tmp_path: Path):
    # train tiny model
    frames = simulate(SimParams(n_bonds=50, n_days=3, providers=["P1"], seed=30, outdir=tmp_path))
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

    # pack
    pkg_dir = tmp_path / "serving/packages"
    res = runner.invoke(pack_app, ["--models-dir", str(tmp_path / "models"), "--run-id", "run", "--outdir", str(pkg_dir)])
    assert res.exit_code == 0, res.stdout
    z = pkg_dir / "run.zip"
    assert z.exists()

    # score a small parquet
    test_rows = feats["test"][feat_cols].head(5)
    in_pq = tmp_path / "in.parquet"; test_rows.to_parquet(in_pq, index=False)
    out_pq = tmp_path / "out.parquet"
    res = runner.invoke(score_app, ["--package", str(z), "--input-path", str(in_pq), "--output-path", str(out_pq), "--device", "cpu"])
    assert res.exit_code == 0, res.stdout
    df = pd.read_parquet(out_pq)
    assert len(df) == len(test_rows)
    assert "preds_bps" in df.columns
