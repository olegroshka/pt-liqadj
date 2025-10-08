from pathlib import Path
import yaml
from typer.testing import CliRunner
from ptliq.cli.run import app as run_app
from ptliq.cli.report import app as report_app

runner = CliRunner(mix_stderr=False)

def test_html_report_generated(tmp_path: Path):
    # tiny config to run fast
    cfg = {
        "project": {"name": "pt-liqadj", "seed": 42, "run_id": "exp_html"},
        "paths": {"raw_dir": "data/raw/sim", "interim_dir": "data/interim",
                  "features_dir": "data/features", "models_dir": "models",
                  "reports_dir": "reports"},
        "data": {"sim": {"n_bonds": 50, "n_days": 3, "providers": ["P1"], "seed": 9}},
        "split": {"train_end": "2025-01-02", "val_end": "2025-01-03"},
        "train": {"device": "cpu", "max_epochs": 2, "batch_size": 256, "lr": 1e-3,
                  "patience": 1, "hidden": [16], "dropout": 0.0, "seed": 99},
    }
    cfg_path = tmp_path / "exp.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    # run the pipeline
    res = runner.invoke(run_app, ["--config", str(cfg_path), "--workdir", str(tmp_path)])
    assert res.exit_code == 0, res.stdout

    # build HTML from latest backtest
    res = runner.invoke(report_app, [
        "--reports-dir", str(tmp_path / "reports"),
        "--run-id", "exp_html",
        "--make-html", "True",
    ])
    assert res.exit_code == 0, res.stdout

    # verify html exists and references our images
    bt_root = tmp_path / "reports" / "exp_html" / "backtest"
    latest = sorted([p for p in bt_root.glob("*") if p.is_dir()])[-1]
    html = latest / "report.html"
    assert html.exists()
    txt = html.read_text(encoding="utf-8")
    assert "Calibration" in txt and "Residuals" in txt
    assert "figures/calibration.png" in txt and "figures/residual_hist.png" in txt
