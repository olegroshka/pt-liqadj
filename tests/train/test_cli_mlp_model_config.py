from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

from ptliq.cli.train_baseline import app_main as mlp_train_main
from ptliq.service.scoring import MLPScorer


def _make_feature_splits(root: Path, run_id: str = "exp_baseline") -> tuple[Path, Path]:
    base = root / "features" / run_id
    base.mkdir(parents=True, exist_ok=True)
    # tiny synthetic dataset: y_bps = 2*f1 - 3*f2 + noise
    rs = np.random.default_rng(0)
    n_train, n_val = 128, 64
    f1_tr = rs.normal(0.0, 1.0, size=n_train)
    f2_tr = rs.normal(0.0, 1.0, size=n_train)
    y_tr = 2.0 * f1_tr - 3.0 * f2_tr + rs.normal(0.0, 0.1, size=n_train)

    f1_va = rs.normal(0.0, 1.0, size=n_val)
    f2_va = rs.normal(0.0, 1.0, size=n_val)
    y_va = 2.0 * f1_va - 3.0 * f2_va + rs.normal(0.0, 0.1, size=n_val)

    df_tr = pd.DataFrame({
        "f_one": f1_tr.astype(float),
        "f_two": f2_tr.astype(float),
        "y_bps": y_tr.astype(float),
    })
    df_va = pd.DataFrame({
        "f_one": f1_va.astype(float),
        "f_two": f2_va.astype(float),
        "y_bps": y_va.astype(float),
    })
    (base / "train.parquet").write_bytes(df_tr.to_parquet(index=False))
    (base / "val.parquet").write_bytes(df_va.to_parquet(index=False))
    return base, base


def test_mlp_persists_and_uses_model_config(tmp_path: Path):
    # 1) create minimal feature splits
    run_id = "exp_baseline"
    features_base, _ = _make_feature_splits(tmp_path, run_id=run_id)

    # 2) train tiny MLP via CLI to produce artifacts
    models_dir = tmp_path / "models"
    mlp_train_main(
        features_dir=features_base.parent,  # pass features root (without run_id)
        run_id=run_id,
        models_dir=models_dir,
        device="cpu",
        max_epochs=2,
        batch_size=64,
        lr=1e-3,
        patience=2,
        seed=7,
        hidden="16,8",
        dropout=0.0,
        config=None,
        workdir=tmp_path,  # ignored when config=None
        verbose=False,
    )

    outdir = models_dir / run_id

    # 3) artifacts: feature_names, scaler, ckpt, train_config, and model_config.json
    assert (outdir / "ckpt.pt").exists(), "checkpoint should be written"
    assert (outdir / "feature_names.json").exists(), "feature_names.json should be written"
    assert (outdir / "scaler.json").exists(), "scaler.json should be written"
    assert (outdir / "train_config.json").exists(), "train_config.json should be written"
    assert (outdir / "model_config.json").exists(), "model_config.json should be written"

    # 4) model_config content sanity
    cfg = json.loads((outdir / "model_config.json").read_text())
    assert cfg.get("model_type") == "mlp"
    assert cfg.get("in_dim") == 2
    assert list(map(int, cfg.get("hidden", []))) == [16, 8]

    # 5) scorer should load and score using that directory
    scorer = MLPScorer.from_dir(outdir, device="cpu")
    rows = [
        {"f_one": 0.5, "f_two": -1.0},
        {"f_one": -0.1, "f_two": 0.4},
    ]
    y = scorer.score_many(rows)
    assert y.shape == (2,)
    assert np.all(np.isfinite(y))
