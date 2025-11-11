import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Usage: python scripts/compute_metrics_fixed.py paper/tables/test_portfolios_eval.csv

path = Path(sys.argv[1] if len(sys.argv) > 1 else 'paper/tables/test_portfolios_eval.csv')
df = pd.read_csv(path)

# Helpers

def metrics(x, y):
    x = pd.to_numeric(x, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')
    m = pd.DataFrame({'x': x, 'y': y}).dropna()
    if len(m) == 0:
        return dict(n=0, r=np.nan, mae=np.nan, rmse=np.nan)
    xv = m['x'].to_numpy()
    yv = m['y'].to_numpy()
    r = float(np.corrcoef(xv, yv)[0,1]) if len(m) > 1 else np.nan
    mae = float(np.mean(np.abs(xv - yv)))
    rmse = float(np.sqrt(np.mean((xv - yv)**2)))
    return dict(n=len(m), r=r, mae=mae, rmse=rmse)

# Row-level pairs
pairs = [
    ("ROW resid WITH pf", 'pred_residual_bps', 'residual_bps'),
    ("ROW resid NO pf",  'pred_residual_bps_nopf', 'residual_bps'),
]
if {'pred_y_bps','truth_y_bps'}.issubset(df.columns):
    pairs += [
        ("ROW y_bps WITH pf", 'pred_y_bps', 'truth_y_bps'),
    ]
    if 'pred_y_bps_nopf' in df.columns:
        pairs += [("ROW y_bps NO pf", 'pred_y_bps_nopf', 'truth_y_bps')]

# New: alternate truths from samples.parquet (training target)
if {'truth_residual_from_samples_bps'}.issubset(df.columns):
    pairs += [
        ("ROW resid WITH pf vs samples.y", 'pred_residual_bps', 'truth_residual_from_samples_bps'),
        ("ROW resid NO pf vs samples.y",  'pred_residual_bps_nopf', 'truth_residual_from_samples_bps'),
    ]
if {'truth_y_from_samples_bps','pred_y_bps'}.issubset(df.columns):
    pairs += [
        ("ROW y_bps WITH pf vs samples.y", 'pred_y_bps', 'truth_y_from_samples_bps'),
    ]
    if 'pred_y_bps_nopf' in df.columns:
        pairs += [("ROW y_bps NO pf vs samples.y", 'pred_y_bps_nopf', 'truth_y_from_samples_bps')]

# Basket-level (mean per portfolio_id+trade_dt)
keys = [k for k in ('portfolio_id','trade_dt') if k in df.columns]
agg_results = []
if keys:
    g = df.groupby(keys)
    agg = {}
    for col in ['pred_residual_bps', 'pred_residual_bps_nopf', 'residual_bps', 'pred_y_bps', 'pred_y_bps_nopf', 'truth_y_bps']:
        if col in df.columns:
            agg[col] = 'mean'
    agg_df = g.agg(agg).reset_index()
    if {'pred_residual_bps','residual_bps'}.issubset(agg_df.columns):
        pairs.append(("BASKET resid WITH pf", 'pred_residual_bps', 'residual_bps'))
    if {'pred_residual_bps_nopf','residual_bps'}.issubset(agg_df.columns):
        pairs.append(("BASKET resid NO pf", 'pred_residual_bps_nopf', 'residual_bps'))
    if {'pred_y_bps','truth_y_bps'}.issubset(agg_df.columns):
        pairs.append(("BASKET y_bps WITH pf", 'pred_y_bps', 'truth_y_bps'))
    if {'pred_y_bps_nopf','truth_y_bps'}.issubset(agg_df.columns):
        pairs.append(("BASKET y_bps NO pf", 'pred_y_bps_nopf', 'truth_y_bps'))
    # New: vs samples truths
    if {'pred_residual_bps','truth_residual_from_samples_bps'}.issubset(agg_df.columns):
        pairs.append(("BASKET resid WITH pf vs samples.y", 'pred_residual_bps', 'truth_residual_from_samples_bps'))
    if {'pred_residual_bps_nopf','truth_residual_from_samples_bps'}.issubset(agg_df.columns):
        pairs.append(("BASKET resid NO pf vs samples.y", 'pred_residual_bps_nopf', 'truth_residual_from_samples_bps'))
    if {'pred_y_bps','truth_y_from_samples_bps'}.issubset(agg_df.columns):
        pairs.append(("BASKET y_bps WITH pf vs samples.y", 'pred_y_bps', 'truth_y_from_samples_bps'))
    if {'pred_y_bps_nopf','truth_y_from_samples_bps'}.issubset(agg_df.columns):
        pairs.append(("BASKET y_bps NO pf vs samples.y", 'pred_y_bps_nopf', 'truth_y_from_samples_bps'))
else:
    agg_df = None

print(f"Metrics from {path} (aligned columns):")
for name, xcol, ycol in pairs:
    if name.startswith('BASKET'):
        src = agg_df
    else:
        src = df
    if src is None or xcol not in src.columns or ycol not in src.columns:
        continue
    m = metrics(src[xcol], src[ycol])
    print(f"{name}: n={m['n']}, r={m['r']:.3f} MAE={m['mae']:.3f} RMSE={m['rmse']:.3f}")
