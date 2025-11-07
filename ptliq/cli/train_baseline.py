from __future__ import annotations
from pathlib import Path
import json
import typer
from rich import print
import numpy as np
import logging

from ptliq.training.dataset import (
    load_split_parquet, discover_features,
    compute_standardizer, apply_standardizer
)
from ptliq.training.loop import TrainConfig, train_loop
from ptliq.utils.config import load_config, get_train_config  # YAML support

app = typer.Typer(no_args_is_help=True)

@app.command()
def app_main(
    features_dir: Path = typer.Option(Path("data/features"), help="Base features dir (ignored when --config is provided)"),
    run_id: str = typer.Option("exp001", help="Run identifier (ignored when --config is provided)"),
    models_dir: Path = typer.Option(Path("models"), help="Where to write model artifacts (ignored when --config is provided)"),
    device: str = typer.Option("cpu", help="cpu|cuda (ignored when --config is provided)"),
    max_epochs: int = typer.Option(20),
    batch_size: int = typer.Option(2048),
    lr: float = typer.Option(1e-3),
    patience: int = typer.Option(3),
    seed: int = typer.Option(42),
    hidden: str = typer.Option("64,64", help="Comma-separated hidden layer sizes"),
    dropout: float = typer.Option(0.0),
    # YAML config inputs
    config: Path | None = typer.Option(None, help="YAML experiment config (uses paths/project/train from this file)"),
    workdir: Path = typer.Option(Path("."), help="Base working directory for resolving paths when using --config"),
    # Logging controls
    verbose: bool = typer.Option(False, help="Enable INFO-level logging (shows epoch summaries and step logs)"),
):
    """
    Train a tiny MLP baseline on y_bps using the featurized parquet files.

    Two modes:
      1) Flag-driven (default): uses command-line options and expects features under --features-dir/--run-id.
      2) YAML-driven: pass --config to read paths and training hyperparameters from the YAML
         (compatible with configs/mlp.default.yaml). In this mode, CLI hyperparameter flags are ignored.
    """
    # Resolve inputs either from YAML or CLI flags
    if config is not None:
        cfg_root = load_config(config)
        # Resolve base dirs under workdir
        features_dir_res = (workdir / cfg_root.paths.features_dir).resolve()
        models_dir_res = (workdir / cfg_root.paths.models_dir).resolve()
        run_id_res = cfg_root.project.run_id
        train_node = get_train_config(cfg_root)

        # Build TrainConfig from YAML node
        hidden_list = list(train_node.hidden or [])
        train_cfg = TrainConfig(
            batch_size=int(train_node.batch_size),
            max_epochs=int(train_node.max_epochs),
            lr=float(train_node.lr),
            patience=int(train_node.patience),
            hidden=hidden_list,
            dropout=float(train_node.dropout),
            device=str(train_node.device),
            seed=int(train_node.seed),
            enable_tqdm=bool(getattr(train_node, "enable_tqdm", False)),
            log_every=int(getattr(train_node, "log_every", 0)),
        )
    else:
        # Fallback to CLI flags mode
        features_dir_res = Path(features_dir)
        models_dir_res = Path(models_dir)
        run_id_res = run_id
        hidden_list = [int(x) for x in hidden.split(",") if x.strip()]
        train_cfg = TrainConfig(
            batch_size=batch_size, max_epochs=max_epochs, lr=lr,
            patience=patience, hidden=hidden_list, dropout=dropout,
            device=device, seed=seed
        )

    # Configure logging based on flags/config
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        level = logging.INFO if verbose or getattr(train_cfg, "log_every", 0) > 0 else logging.WARNING
        logging.basicConfig(level=level, format="[%(levelname)s] %(name)s: %(message)s")
    # Always ensure our training logger is at least INFO when verbose
    if verbose:
        logging.getLogger("ptliq.training.mlp").setLevel(logging.INFO)
        # Ensure existing handlers don't filter out INFO
        for h in root_logger.handlers:
            try:
                h.setLevel(logging.INFO)
            except Exception:
                pass

    # Load feature splits
    base = Path(features_dir_res) / run_id_res
    train_df = load_split_parquet(base, "train")
    val_df = load_split_parquet(base, "val")

    feat_cols = discover_features(train_df)
    if not feat_cols:
        raise typer.BadParameter("No features found (columns starting with 'f_').")

    # standardize
    stdz = compute_standardizer(train_df, feat_cols)
    X_train = apply_standardizer(train_df, feat_cols, stdz)
    X_val = apply_standardizer(val_df, feat_cols, stdz)
    y_train = train_df["y_bps"].astype(float).to_numpy()
    y_val = val_df["y_bps"].astype(float).to_numpy()

    # train
    model_out = Path(models_dir_res) / run_id_res
    model_out.mkdir(parents=True, exist_ok=True)

    res = train_loop(X_train, y_train, X_val, y_val, feat_cols, model_out, train_cfg)

    # persist scaler + cfg
    with open(model_out / "scaler.json", "w", encoding="utf-8") as f:
        json.dump({"mean": stdz["mean"].tolist(), "std": stdz["std"].tolist()}, f, indent=2)
    with open(model_out / "train_config.json", "w", encoding="utf-8") as f:
        # Prefer persisting the YAML-derived node when available; else dump TrainConfig
        if config is not None:
            json.dump(train_cfg.__dict__, f, indent=2)
        else:
            json.dump(train_cfg.__dict__, f, indent=2)

    print(f"[bold green]TRAIN OK[/bold green] → {model_out}")
    print(f"  best_epoch={res['best_epoch']}  val_mae_bps={res['best_val_mae_bps']:.3f}")

# expose typer app
app = app

if __name__ == "__main__":
    app()
