from __future__ import annotations
from pathlib import Path
import json
import typer
from rich import print
import numpy as np

from ptliq.training.dataset import (
    load_split_parquet, discover_features,
    compute_standardizer, apply_standardizer
)
from ptliq.training.loop import TrainConfig, train_loop

app = typer.Typer(no_args_is_help=True)

@app.command()
def app_main(
    features_dir: Path = typer.Option(Path("data/features"), help="Base features dir"),
    run_id: str = typer.Option("exp001", help="Run identifier (subfolder under features_dir and models_dir)"),
    models_dir: Path = typer.Option(Path("models"), help="Where to write model artifacts"),
    device: str = typer.Option("cpu", help="cpu|cuda"),
    max_epochs: int = typer.Option(20),
    batch_size: int = typer.Option(2048),
    lr: float = typer.Option(1e-3),
    patience: int = typer.Option(3),
    seed: int = typer.Option(42),
    hidden: str = typer.Option("64,64", help="Comma-separated hidden layer sizes"),
    dropout: float = typer.Option(0.0),
):
    """
    Train a tiny MLP baseline on y_bps using the featurized parquet files.
    """
    base = Path(features_dir) / run_id
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
    model_out = Path(models_dir) / run_id
    model_out.mkdir(parents=True, exist_ok=True)

    hidden_list = [int(x) for x in hidden.split(",") if x.strip()]
    cfg = TrainConfig(
        batch_size=batch_size, max_epochs=max_epochs, lr=lr,
        patience=patience, hidden=hidden_list, dropout=dropout,
        device=device, seed=seed
    )
    res = train_loop(X_train, y_train, X_val, y_val, feat_cols, model_out, cfg)

    # persist scaler + cfg
    with open(model_out / "scaler.json", "w", encoding="utf-8") as f:
        json.dump({"mean": stdz["mean"].tolist(), "std": stdz["std"].tolist()}, f, indent=2)
    with open(model_out / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, indent=2)

    print(f"[bold green]TRAIN OK[/bold green] → {model_out}")
    print(f"  best_epoch={res['best_epoch']}  val_mae_bps={res['best_val_mae_bps']:.3f}")

# expose typer app
app = app

if __name__ == "__main__":
    app()
