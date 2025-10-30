# ptliq/cli/run.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import json
import os
import typer
from rich import print
import torch

from ptliq.utils.config import load_config, get_sim_config, get_split_config, get_train_config
from ptliq.data.simulate import SimParams, simulate
from ptliq.data.validate import validate_raw
from ptliq.data.split import compute_default_ranges, write_ranges
from ptliq.features.build import build_features
from ptliq.training.dataset import discover_features, compute_standardizer, apply_standardizer
from ptliq.training.loop import TrainConfig as TrainCfg, train_loop
from ptliq.backtest.protocol import run_backtest
from ptliq.viz.report import render_report

# MV-DGT pieces
from ptliq.cli.featurize import featurize_graph as _featurize_graph, featurize_pyg as _featurize_pyg
from ptliq.features.build_mvdgt_dataset import build as _mvdgt_build
from ptliq.training.mvdgt_loop import MVDGTTrainConfig as _MVDGTTrainCfg, train_mvdgt as _mvdgt_train

app = typer.Typer(no_args_is_help=True)


def _echo_cmd(cmd: str):
    print(f"[cyan]$ {cmd}[/cyan]")


@app.command()
def app_main(
    config: Path = typer.Option(..., help="YAML experiment config"),
    workdir: Path = typer.Option(Path("."), help="Base working directory"),
):
    """
    Run pipeline.
    - Default: baseline simulate → validate → split → featurize → train → backtest → report.
    - If config contains data.mvdgt section: run MV-DGT end-to-end steps and echo each command.
    All artifacts go under <workdir>/{data/features,models,reports}/<run_id>/... (baseline)
    and <workdir>/{data,models}/mvdgt/<run_id>... (MV-DGT) where applicable.
    """
    cfg = load_config(config)

    # If config requests MV-DGT pipeline, run it and return early
    mvdgt_cfg = None
    try:
        # place under data.mvdgt in YAML to avoid changing RootConfig
        mvdgt_cfg = cfg.data.get("mvdgt")
    except Exception:
        mvdgt_cfg = None

    if mvdgt_cfg:
        # resolve common dirs
        run_id = cfg.project.run_id
        # default dirs (can be overridden via env in underlying CLIs; we resolve here explicitly)
        rawdir = (workdir / cfg.paths.raw_dir).resolve()
        graph_dir = Path(mvdgt_cfg.get("graph_dir", os.getenv("PTLIQ_DEFAULT_GRAPH_DIR", "data/graph")))
        graph_dir = (workdir / graph_dir).resolve()
        pyg_dir = Path(mvdgt_cfg.get("pyg_dir", os.getenv("PTLIQ_DEFAULT_PYG_DIR", "data/pyg")))
        pyg_dir = (workdir / pyg_dir).resolve()
        mvdgt_out = Path(mvdgt_cfg.get("outdir", f"data/mvdgt/{run_id}"))
        mvdgt_out = (workdir / mvdgt_out).resolve()
        models_dir = Path(mvdgt_cfg.get("models_dir", os.getenv("PTLIQ_DEFAULT_MODELS_DIR", "models/mvdgt")))
        models_dir = (workdir / models_dir).resolve()

        # ensure dirs exist
        rawdir.mkdir(parents=True, exist_ok=True)
        graph_dir.mkdir(parents=True, exist_ok=True)
        pyg_dir.mkdir(parents=True, exist_ok=True)
        mvdgt_out.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)

        # Optional simulate (if requested) to produce raw bonds/trades
        sim_section = mvdgt_cfg.get("sim") or cfg.data.get("sim")
        if sim_section:
            sim = get_sim_config(cfg) if "sim" in cfg.data else None
            # If a custom sim section is provided under mvdgt, override
            if isinstance(sim_section, dict):
                from pydantic import BaseModel
                class _S(BaseModel):
                    n_bonds: int = 200
                    n_days: int = 10
                    providers: list[str] = ["P1","P2"]
                    seed: int = 42
                s = _S(**sim_section)
                sim_params = SimParams(n_bonds=s.n_bonds, n_days=s.n_days, providers=s.providers, seed=s.seed, outdir=rawdir)
            else:
                sim = get_sim_config(cfg)
                sim_params = SimParams(n_bonds=sim.n_bonds, n_days=sim.n_days, providers=sim.providers, seed=sim.seed, outdir=rawdir)

            _echo_cmd(f"ptliq-simulate --outdir {rawdir}")
            frames = simulate(sim_params)
            frames["bonds"].to_parquet(rawdir / "bonds.parquet", index=False)
            frames["trades"].to_parquet(rawdir / "trades.parquet", index=False)
            print(f"[bold]SIM[/bold] → {rawdir}")

        # Validate raw tables if present
        if (rawdir/"bonds.parquet").exists() and (rawdir/"trades.parquet").exists():
            _echo_cmd(f"ptliq-validate --rawdir {rawdir}")
            report = validate_raw(rawdir)
            stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
            val_dir = (workdir / cfg.paths.interim_dir / "validated").resolve()
            val_dir.mkdir(parents=True, exist_ok=True)
            (val_dir / f"validation_report_{stamp}.json").write_text(json.dumps(report, indent=2))
            if not report.get("passed", True):
                print("[yellow]Validation did not pass — continuing.[/yellow]")

        # Build graph from raw if needed
        bonds_p = rawdir/"bonds.parquet"
        trades_p = rawdir/"trades.parquet"
        if bonds_p.exists() and trades_p.exists():
            _echo_cmd(f"ptliq-featurize featurize-graph --bonds {bonds_p} --trades {trades_p} --outdir {graph_dir}")
            # Call with explicit concrete values for all options to avoid Typer OptionInfo defaults
            _featurize_graph(
                bonds=bonds_p,
                trades=trades_p,
                outdir=graph_dir,
                expo_scale=5e4,
                target_min=200,
                target_max=300,
                cotrade_q=0.85,
                cotrade_topk=20,
                issuer_topk=20,
                sector_topk=8,
                rating_topk=4,
                curve_topk=4,
                currency_topk=0,
                corr_enable=True,
                corr_pcc=True,
                corr_mi=True,
                corr_local=True,
                corr_global=True,
                corr_topk=20,
                corr_local_days=20,
                progress=True,
            )

        # Build PyG artifacts
        _echo_cmd(f"ptliq-featurize featurize-pyg --graph-dir {graph_dir} --outdir {pyg_dir}")
        _featurize_pyg(graph_dir=graph_dir, outdir=pyg_dir)

        # Build MV-DGT dataset
        split_train = float(mvdgt_cfg.get("split_train", 0.70))
        split_val = float(mvdgt_cfg.get("split_val", 0.15))
        trades_path = Path(mvdgt_cfg.get("trades_path", str(trades_p if trades_p.exists() else rawdir/"trades.parquet")))
        _echo_cmd(f"ptliq-dgt-build --trades-path {trades_path} --graph-dir {graph_dir} --pyg-dir {pyg_dir} --outdir {mvdgt_out} --split-train {split_train} --split-val {split_val}")
        _mvdgt_build(trades_path=trades_path, graph_dir=graph_dir, pyg_dir=pyg_dir, outdir=mvdgt_out, split_train=split_train, split_val=split_val)
        print(f"[bold]MVDGT-BUILD[/bold] → {mvdgt_out}")

        # Train MV-DGT
        epochs = int(mvdgt_cfg.get("epochs", 5))
        lr = float(mvdgt_cfg.get("lr", 1e-3))
        weight_decay = float(mvdgt_cfg.get("weight_decay", 1e-4))
        batch_size = int(mvdgt_cfg.get("batch_size", 512))
        seed = int(mvdgt_cfg.get("seed", 17))
        device_opt = mvdgt_cfg.get("device", "auto")
        device_str = ("cuda" if torch.cuda.is_available() else "cpu") if str(device_opt).lower() == "auto" else str(device_opt)

        # models dir via env for trainer default
        os.environ["PTLIQ_DEFAULT_MODELS_DIR"] = str(models_dir)
        _echo_cmd(f"ptliq-dgt-train --workdir {mvdgt_out} --pyg-dir {pyg_dir} --outdir {models_dir} --epochs {epochs} --lr {lr} --weight-decay {weight_decay} --batch-size {batch_size} --seed {seed} --device-str {device_str}")
        metrics = _mvdgt_train(_MVDGTTrainCfg(
            workdir=mvdgt_out,
            pyg_dir=pyg_dir,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            seed=seed,
            device=device_str,
            tb_log_dir=str(models_dir/"tb"),
        ))
        # persist metrics
        (models_dir/"mvdgt_metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"[bold]MVDGT-TRAIN[/bold] → {models_dir}  best_val_mae_bps={metrics.get('best_val_mae_bps','?')}")
        return

    # ------------------------
    # Default BASELINE pipeline
    # ------------------------
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
        print(f"[bold yellow]VALIDATION DID NOT PASS[/bold yellow] → continuing. See {report_path}")
    else:
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
