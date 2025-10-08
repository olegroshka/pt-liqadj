from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

def plot_calibration(metrics_json: Path, out_png: Path) -> Path:
    with open(metrics_json, "r", encoding="utf-8") as f:
        m = json.load(f)
    calib = pd.DataFrame(m["calibration"])
    if calib.empty:
        return out_png
    plt.figure()
    plt.plot(calib["pred_mean"], calib["y_mean"], marker="o", linestyle="-", label="binned means")
    # reference line
    lo = float(min(calib["pred_mean"] + calib["y_mean"]))
    hi = float(max(calib["pred_mean"] + calib["y_mean"]))
    # safer bounds from data
    lo = float(min(calib["pred_mean"] + calib["y_mean"]))
    plt.plot(calib["pred_mean"], calib["pred_mean"], linestyle="--", label="ideal")
    plt.xlabel("Predicted (bps)")
    plt.ylabel("Observed mean (bps)")
    plt.title("Calibration (binned by prediction)")
    plt.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    return out_png
