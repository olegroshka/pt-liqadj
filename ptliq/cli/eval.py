# ptliq/cli/eval.py
from __future__ import annotations
from pathlib import Path
import json
import typer
from rich import print
import numpy as np
import torch

from ptliq.training.dataset import load_split_parquet, discover_features, apply_standardizer
from ptliq.training.loop import load_model_for_eval

app = typer.Typer(no_args_is_help=True)

@app.command()
def app_main(
    features_dir: Path = typer.Option(Path("data/features")),
    run_id: str = typer.Option("exp001"),
    models_dir: Path = typer.Option(Path("models")),
    device: str = typer.Option("cpu"),
):
    """
    Evaluate the trained model on test.parquet and write metrics_test.json.
    """
    base = Path(features_dir) / run_id
    test_df = load_split_parquet(base, "test")
    feat_cols = discover_features(test_df)
    if not feat_cols:
        raise typer.BadParameter("No features found (columns starting with 'f_').")

    mdir = Path(models_dir) / run_id
    with open(mdir / "scaler.json", "r", encoding="utf-8") as f:
        stdz = json.load(f)
    stdz = {"mean": np.array(stdz["mean"], dtype=np.float32), "std": np.array(stdz["std"], dtype=np.float32)}

    X_test = apply_standardizer(test_df, feat_cols, stdz)
    y_test = test_df["y_bps"].astype(float).to_numpy().astype(np.float32)

    # read train_config to get the correct architecture
    hidden = [64, 64]
    dropout = 0.0
    cfg_path = mdir / "train_config.json"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            tr_cfg = json.load(f)
        hidden = tr_cfg.get("hidden", hidden)
        dropout = tr_cfg.get("dropout", dropout)

    model, dev = load_model_for_eval(mdir, in_dim=X_test.shape[1], hidden=hidden, dropout=dropout, device=device)
    with torch.no_grad():
        y_pred = model(torch.from_numpy(X_test).to(dev)).cpu().numpy()

    err = y_pred - y_test
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))

    out = {"mae_bps": mae, "rmse_bps": rmse, "n": int(len(y_test))}
    with open(mdir / "metrics_test.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"[bold green]EVAL OK[/bold green]  n={out['n']}  test_mae_bps={mae:.3f}  rmse_bps={rmse:.3f}")

# expose Typer app
app = app

if __name__ == "__main__":
    app()
