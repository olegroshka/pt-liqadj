from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

def basic_metrics(y: np.ndarray, yhat: np.ndarray) -> Dict[str, float]:
    err = yhat - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    bias = float(np.mean(err))
    return {"mae_bps": mae, "rmse_bps": rmse, "bias_bps": bias, "n": int(len(y))}

def slice_by_column(df: pd.DataFrame, col: str, y_col: str = "y_bps", yhat_col: str = "yhat_bps") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, grp in df.groupby(col):
        m = basic_metrics(grp[y_col].to_numpy(), grp[yhat_col].to_numpy())
        out[str(key)] = m
    return out

def decile_slices(df: pd.DataFrame, col: str, y_col: str = "y_bps", yhat_col: str = "yhat_bps") -> Dict[str, Any]:
    # rank into ~10 equal-sized bins (robust quantile binning)
    q = pd.qcut(df[col].rank(method="first"), 10, labels=False, duplicates="drop")
    df = df.assign(_decile=q)
    return slice_by_column(df, "_decile", y_col, yhat_col)

def calibration_bins(y: np.ndarray, yhat: np.ndarray, n_bins: int = 15) -> pd.DataFrame:
    # bin by prediction; plot mean(y) vs mean(pred)
    idx = np.argsort(yhat)
    yhat_s = yhat[idx]; y_s = y[idx]
    bins = np.array_split(np.arange(len(yhat_s)), min(n_bins, max(1, len(yhat_s))))
    rows = []
    for i, b in enumerate(bins):
        if len(b) == 0:
            continue
        phat = float(np.mean(yhat_s[b]))
        ybar = float(np.mean(y_s[b]))
        rows.append({"bin": i, "pred_mean": phat, "y_mean": ybar, "n": int(len(b))})
    return pd.DataFrame(rows)
