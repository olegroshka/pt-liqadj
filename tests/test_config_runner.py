# tests/test_config_runner.py
from pathlib import Path
import yaml
from typer.testing import CliRunner
from ptliq.cli.run import app as run_app

runner = CliRunner(mix_stderr=False)

def test_config_run_e2e(tmp_path: Path):
    cfg = {
        "project": {"name": "pt-liqadj", "seed": 42, "run_id": "exp_cfg"},
        "paths": {
            "raw_dir": "data/raw/sim",
            "interim_dir": "data/interim",
            "features_dir": "data/features",
            "models_dir": "models",
            "reports_dir": "reports",
        },
        "data": {"sim": {"n_bonds": 60, "n_days": 3, "providers": ["P1"], "seed": 10}},
        "split": {"train_end": "2025-01-02", "val_end": "2025-01-03"},
        "train": {"device": "cpu", "max_epochs": 2, "batch_size": 256, "lr": 1e-3, "patience": 1, "hidden": [16], "dropout": 0.0, "seed": 99},
    }
    cfg_path = tmp_path / "exp.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    res = runner.invoke(run_app, ["--config", str(cfg_path), "--workdir", str(tmp_path)])
    assert res.exit_code == 0, res.stdout

    # artifacts exist
    feat_dir = tmp_path / "data" / "features" / "exp_cfg"
    assert (feat_dir / "train.parquet").exists()
    mdir = tmp_path / "models" / "exp_cfg"
    assert (mdir / "ckpt.pt").exists() and (mdir / "train_config.json").exists()
    rbase = tmp_path / "reports" / "exp_cfg" / "backtest"
    stamps = list(rbase.glob("*"))
    assert stamps, "no backtest stamp dir"
    latest = sorted(stamps)[-1]
    assert (latest / "metrics.json").exists()
    assert (latest / "figures" / "calibration.png").exists()
