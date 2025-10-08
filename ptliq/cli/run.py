# ptliq/cli/run.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import json
import typer
from rich import print

from ptliq.utils.config import load_config, get_sim_config, get_split_config, get_train_config
from ptliq.data.simulate import SimParams, simulate
from ptliq.data.validate import validate_raw
from ptliq.data.split import compute_default_ranges, write_ranges
from ptliq.features.build import build_features
from ptliq.training.dataset import discover_features, compute_standardizer, apply_standardizer
from ptliq.training.loop import TrainConfig as TrainCfg, train_loop
from ptliq.backtest.protocol import run_backtest
from ptliq.viz.report import render_report

app = typer.Typer(no_args_is_help=True)

@app.command()
def app_main(
    config: Path = typer.Option(..., help="YAML experiment config"),
    workdir: Path = typer.Option(Path("."), help="Base working directory"),
):
    """
    Run: simulate → validate → split → featurize → train → backtest → report.
    All artifacts go under <workdir>/{data/features,models,reports}/<run_id>/...
    """
    cfg = load_config(config)
    sim = get_sim_config(cfg)
    split = get_split_config(cfg)
    train_node = get_train_config(cfg)

    # resolve paths under workdir
    rawdir     = (workdir / cfg.paths.raw_dir).resolve()
    interim    = (workdir / cfg.paths.interim_dir).resolve()
    features_d = (workdir / cfg.paths.features_dir).resolve()
    models_d   = (workdir / cfg.paths.models_dir).resolve()
    reports_d  = (workdir / cfg.paths.reports_dir).resolve()
    run_id = cfg.project.run_id

    rawdir.mkdir(parents=True, exist_ok=True)
    interim.mkdir(parents=True, exist_ok=True)
    (features_d / run_id).mkdir(parents=True, exist_ok=True)
    (models_d / run_id).mkdir(parents=True, exist_ok=True)
    (reports_d / run_id).mkdir(parents=True, exist_ok=True)

    # 0) simulate
    frames = simulate(SimParams(
        n_bonds=sim.n_bonds, n_days=sim.n_days, providers=sim.providers, seed=sim.seed, outdir=rawdir
    ))
    frames["bonds"].to_parquet(rawdir / "bonds.parquet", index=False)
    frames["trades"].to_parquet(rawdir / "trades.parquet", index=False)
    print(f"[bold]SIM[/bold] → {rawdir}")

    # 1) validate
    val_report_dir = interim / "validated"
    val_report_dir.mkdir(parents=True, exist_ok=True)

    report = validate_raw(rawdir)  # <-- no outdir kwarg
    # persist a JSON report (mimic the CLI behavior)
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    report_path = val_report_dir / f"validation_report_{stamp}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if not report["passed"]:
        raise RuntimeError(f"Validation failed; see {report_path}")

    print(f"[bold]VAL[/bold] passed → {report_path}")

    # 2) split
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    split_dir = interim / "splits" / stamp
    split_dir.mkdir(parents=True, exist_ok=True)
    ranges = compute_default_ranges(rawdir / "trades.parquet", split.train_end, split.val_end)
    rpath = write_ranges(ranges, split_dir)
    print(f"[bold]SPLIT[/bold] → {rpath}")

    # 3) features
    feats = build_features(rawdir, rpath)
    feat_out = features_d / run_id
    for name, df in feats.items():
        df.to_parquet(feat_out / f"{name}.parquet", index=False)
    print(f"[bold]FEAT[/bold] → {feat_out}")

    # 4) train
    train_df, val_df = feats["train"], feats["val"]
    feat_cols = discover_features(train_df)
    if not feat_cols:
        raise RuntimeError("no features found (columns starting with f_)")
    stdz = compute_standardizer(train_df, feat_cols)
    Xtr = apply_standardizer(train_df, feat_cols, stdz)
    Xva = apply_standardizer(val_df,  feat_cols, stdz)
    ytr = train_df["y_bps"].to_numpy()
    yva = val_df["y_bps"].to_numpy()

    model_dir = models_d / run_id
    res = train_loop(
        Xtr, ytr, Xva, yva, feat_cols, model_dir,
        TrainCfg(
            batch_size=train_node.batch_size, max_epochs=train_node.max_epochs,
            lr=train_node.lr, patience=train_node.patience, hidden=train_node.hidden,
            dropout=train_node.dropout, device=train_node.device, seed=train_node.seed
        ),
    )
    # persist scaler + feature names + cfg
    (model_dir / "scaler.json").write_text(json.dumps({"mean": stdz["mean"].tolist(), "std": stdz["std"].tolist()}, indent=2))
    (model_dir / "feature_names.json").write_text(json.dumps(feat_cols, indent=2))
    (model_dir / "train_config.json").write_text(json.dumps(train_node.model_dump(), indent=2))
    print(f"[bold]TRAIN[/bold] → {model_dir} (val_mae_bps={res['best_val_mae_bps']:.3f})")

    # 5) backtest + report
    bt_dir = reports_d / run_id / "backtest" / stamp
    metrics = run_backtest(features_d, run_id, models_d, bt_dir)
    figs = render_report(bt_dir)
    from ptliq.viz.report import render_html
    html = render_html(bt_dir, title=f"{run_id} backtest")
    print(f"  • html: {html}")
    print(f"[bold]BACKTEST[/bold] → {bt_dir}  n={metrics['n']}")
    print(f"[bold]REPORT[/bold] → {bt_dir/'figures'}")
    for k, v in figs.items():
        print(f"  • {k}: {v}")

# expose Typer app
app = app

if __name__ == "__main__":
    app()
