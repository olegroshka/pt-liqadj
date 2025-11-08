
# gru_train.py
"""
CLI wrapper around the minimal GRU baseline defined in gru_loop.py

Usage (example):
  python -m gru_train --feature-dir data/features/gru_baseline --outdir models/gru/run1 \
    --epochs 30 --lr 1e-3 --batch-size 2048 --window 5

You can also provide a YAML --config with optional 'train' and/or 'model' sections.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import typer

from ptliq.training.gru_loop import GRUTrainConfig, GRUModelConfig, train_gru

try:
    import yaml  # optional
except Exception:  # pragma: no cover
    yaml = None


app = typer.Typer(add_completion=False, no_args_is_help=True)



@app.command("train")
def main(
    feature_dir: Path = typer.Option(Path("data/features/gru_baseline"), help="Directory with train/val/test/market_features parquet files."),
    outdir: Path = typer.Option(Path("models/gru/exp_default"), help="Output directory for checkpoints/metrics."),
    device: str = typer.Option("auto", help="'auto', 'cpu', or 'cuda'"),
    # optimization
    epochs: int = typer.Option(30, help="Max epochs"),
    lr: float = typer.Option(1e-3, help="Learning rate"),
    batch_size: int = typer.Option(512, help="Batch size"),
    patience: int = typer.Option(4, help="Early stopping patience (val MAE); set 0 to disable"),
    seed: int = typer.Option(42, help="Random seed"),
    no_early_stop: bool = typer.Option(False, "--no-early-stop", help="Disable early stopping and run all epochs"),
    # model
    hidden: int = typer.Option(64, help="GRU hidden size"),
    layers: int = typer.Option(1, help="Number of GRU layers"),
    dropout: float = typer.Option(0.1, help="Dropout"),
    window: int = typer.Option(5, help="Market lookback window length (days)"),
    # optional YAML config
    config: Path = typer.Option(None, "--config", "-c", help="Optional YAML file with 'train'/'model' sections"),
):
    # base cfg
    cfg = GRUTrainConfig(
        feature_dir=str(feature_dir),
        outdir=str(outdir),
        device=device,
        epochs=int(epochs),
        lr=float(lr),
        batch_size=int(batch_size),
        patience=int(patience),
        seed=int(seed),
        model=GRUModelConfig(hidden=int(hidden), layers=int(layers), dropout=float(dropout), window=int(window)),
    )

    # optional config override
    if config is not None and config.exists():
        if yaml is None:
            raise RuntimeError("PyYAML is not installed; remove --config or install pyyaml")
        y = yaml.safe_load(config.read_text()) or {}
        if "train" in y:
            for k, v in y["train"].items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        if "model" in y:
            for k, v in y["model"].items():
                if hasattr(cfg.model, k):
                    setattr(cfg.model, k, v)

    print("\n[GRU BASELINE] Configuration:\n")
    # Dataclasses inside dataclasses are not directly serializable by asdict twice
    print(json.dumps({"train": asdict(cfg) | {"model": asdict(cfg.model)}}, indent=2))

    # honor --no-early-stop flag
    if no_early_stop:
        cfg.patience = 0

    metrics = train_gru(cfg)

    print("\n[GRU BASELINE] Best validation metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    app()