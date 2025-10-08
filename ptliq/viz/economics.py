from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def plot_residual_hist(residuals_parquet: Path, out_png: Path) -> Path:
    df = pd.read_parquet(residuals_parquet, columns=["y_bps", "yhat_bps"])
    err = df["yhat_bps"] - df["y_bps"]
    plt.figure()
    plt.hist(err, bins=30)
    plt.xlabel("Prediction error (bps)")
    plt.ylabel("Count")
    plt.title("Residual distribution")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    return out_png
