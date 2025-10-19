from pathlib import Path
from typer.testing import CliRunner
import json
import pandas as pd
import glob

from ptliq.cli.simulate import app as sim_app
from ptliq.cli.validate import app as val_app
from ptliq.cli.split import app as split_app
from ptliq.cli.featurize import app as feat_app
from ptliq.cli.train import app as train_app
from ptliq.cli.eval import app as eval_app
from ptliq.cli.backtest import app as bt_app
from ptliq.cli.report import app as rpt_app

runner = CliRunner(mix_stderr=False)

def test_cli_end_to_end_smoke(tmp_path: Path):
    rawdir = tmp_path / "data" / "raw" / "sim"
    splits_dir = tmp_path / "data" / "interim" / "splits"
    feat_base = tmp_path / "data" / "features"
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    cfg = tmp_path / "base.yaml"
    cfg.write_text(
        "project:\n  name: pt-liqadj\n  seed: 42\n"
        "paths:\n  raw_dir: data/raw\n  interim_dir: data/interim\n  features_dir: data/features\n  models_dir: models\n  reports_dir: reports\n"
        "data:\n  sim:\n    n_bonds: 80\n    n_days: 4\n    providers: ['P1','P2']\n    seed: 7\n"
    )

    # simulate
    res = runner.invoke(sim_app, ["--config", str(cfg), "--outdir", str(rawdir), "--seed", "123", "--loglevel", "INFO"])
    assert res.exit_code == 0, res.stdout

    # validate
    res = runner.invoke(val_app, ["--rawdir", str(rawdir), "--outdir", str(tmp_path / "validated"), "--loglevel", "INFO", "--no-fail-on-error"])
    assert res.exit_code == 0, res.stdout

    # split (use 4 sim days starting 2025-01-02 → 02..05 ; train 02..03, val=04, test=05)
    res = runner.invoke(split_app, [
        "--rawdir", str(rawdir),
        "--outdir", str(splits_dir),
        "--train-end", "2025-01-03",
        "--val-end", "2025-01-04",
    ])
    assert res.exit_code == 0, res.stdout
    # new behavior: ranges.json is written directly to the specified outdir
    ranges_json = splits_dir / "ranges.json"
    assert ranges_json.exists(), "ranges.json not created in splits_dir"

    # featurize
    run_id = "exp_cli"
    res = runner.invoke(feat_app, [
        "--rawdir", str(rawdir),
        "--splits", str(ranges_json),
        "--outdir", str(feat_base),
        "--run-id", run_id,
    ])
    assert res.exit_code == 0, res.stdout
    for split in ["train", "val", "test"]:
        assert (feat_base / run_id / f"{split}.parquet").exists()

    # train
    res = runner.invoke(train_app, [
        "--features-dir", str(feat_base),
        "--run-id", run_id,
        "--models-dir", str(models_dir),
        "--device", "cpu",
        "--max-epochs", "3",
        "--batch-size", "512",
        "--lr", "1e-3",
        "--patience", "2",
        "--hidden", "32",
        "--dropout", "0.0",
    ])
    assert res.exit_code == 0, res.stdout
    mdir = models_dir / run_id
    assert (mdir / "ckpt.pt").exists()
    assert (mdir / "scaler.json").exists()
    assert (mdir / "feature_names.json").exists()

    # eval
    res = runner.invoke(eval_app, [
        "--features-dir", str(feat_base),
        "--run-id", run_id,
        "--models-dir", str(models_dir),
        "--device", "cpu",
    ])
    assert res.exit_code == 0, res.stdout
    assert (mdir / "metrics_test.json").exists()
    js = json.loads((mdir / "metrics_test.json").read_text())
    assert js["n"] > 0 and "mae_bps" in js

    # backtest + report
    res = runner.invoke(bt_app, [
        "--features-dir", str(feat_base),
        "--run-id", run_id,
        "--models-dir", str(models_dir),
        "--reports-dir", str(reports_dir),
    ])
    assert res.exit_code == 0, res.stdout
    bt_dirs = sorted((reports_dir / run_id / "backtest").glob("*"))
    assert bt_dirs, "no backtest output folder"
    latest = bt_dirs[-1]
    assert (latest / "metrics.json").exists()
    assert (latest / "residuals.parquet").exists()

    res = runner.invoke(rpt_app, ["--reports-dir", str(reports_dir), "--run-id", run_id])
    assert res.exit_code == 0, res.stdout
    figs_dir = latest / "figures"
    assert (figs_dir / "calibration.png").exists()
    assert (figs_dir / "residual_hist.png").exists()
