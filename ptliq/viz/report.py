from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional
import json
from string import Template

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _load_residuals(backtest_dir: Path) -> Optional[np.ndarray]:
    """
    Try to find residuals in common files written by backtest.
    Returns an array of residuals in bps, or None if not found.
    """
    backtest_dir = Path(backtest_dir)
    candidates = [
        backtest_dir / "residuals.parquet",
        backtest_dir / "residuals.pq",
        backtest_dir / "residuals.csv",
        backtest_dir / "preds.parquet",
        backtest_dir / "predictions.parquet",
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            if p.suffix.lower() in {".parquet", ".pq"}:
                df = pd.read_parquet(p)
            else:
                df = pd.read_csv(p)
        except Exception:
            continue

        cols = {c.lower(): c for c in df.columns}
        # direct residual column
        for k in ("resid_bps", "residual_bps", "residual"):
            if k in cols:
                a = df[cols[k]].to_numpy(dtype=float)
                return a
        # compute y - pred if present
        ycol = cols.get("y_bps") or cols.get("y") or cols.get("target")
        pcol = cols.get("pred_bps") or cols.get("pred") or cols.get("prediction")
        if ycol and pcol:
            a = (df[cols[ycol]].to_numpy(dtype=float)
                 - df[cols[pcol]].to_numpy(dtype=float))
            return a
    return None


def render_report(backtest_dir: Path) -> Dict[str, Path]:
    """
    Create two figures under <backtest_dir>/figures:
      - calibration.png : mean(pred) vs mean(y) per calibration bin + y=x reference
      - residual_hist.png: histogram of residuals (if available) or an approximation

    Returns a dict of figure names to paths.
    """
    backtest_dir = Path(backtest_dir)
    figs = backtest_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    metrics_path = backtest_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found in {backtest_dir}")

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # ---- Calibration plot ----
    cal = metrics.get("calibration") or {}
    bins = np.asarray(cal.get("bin", []), dtype=float)
    pred_mean = np.asarray(cal.get("pred_mean", []), dtype=float)
    y_mean = np.asarray(cal.get("y_mean", []), dtype=float)

    cal_path = figs / "calibration.png"
    if bins.size and pred_mean.size and pred_mean.size == y_mean.size:
        x = pred_mean
        y = y_mean
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        plt.figure()
        plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
        plt.scatter(x, y)
        plt.xlabel("Mean prediction (bps)")
        plt.ylabel("Mean observed (bps)")
        plt.title("Calibration")
        plt.tight_layout()
        plt.savefig(cal_path, dpi=150)
        plt.close()
    else:
        plt.figure()
        plt.title("Calibration (no data)")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(cal_path, dpi=150)
        plt.close()

    # ---- Residual histogram ----
    resid = _load_residuals(backtest_dir)
    if resid is None:
        # fallback: approximate residuals from calibration deltas weighted by n
        cal_n = np.asarray(cal.get("n", []), dtype=float)
        if pred_mean.size and y_mean.size and cal_n.size == pred_mean.size:
            delta = (y_mean - pred_mean)
            reps = np.clip(cal_n.astype(int), 1, None)
            resid = np.repeat(delta, reps)
        else:
            resid = np.array([], dtype=float)

    hist_path = figs / "residual_hist.png"
    plt.figure()
    if resid.size:
        plt.hist(resid, bins=20)
        plt.xlabel("Residual (bps)")
        plt.ylabel("Count")
        plt.title("Residuals histogram")
    else:
        plt.title("Residuals histogram (no data)")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(hist_path, dpi=150)
    plt.close()

    return {"calibration": cal_path, "residual_hist": hist_path}


def render_html(backtest_dir: Path, title: str = "PT-LiqAdj Backtest") -> Path:
    """
    Self-contained HTML report referencing figures and summarizing metrics.json.
    """
    backtest_dir = Path(backtest_dir)
    metrics_path = backtest_dir / "metrics.json"
    figs_dir = backtest_dir / "figures"
    cal_png = figs_dir / "calibration.png"
    hist_png = figs_dir / "residual_hist.png"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json in {backtest_dir}")
    # ensure figures exist (idempotent)
    if not (cal_png.exists() and hist_png.exists()):
        render_report(backtest_dir)

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    overall = metrics.get("overall", {})
    n = overall.get("n", 0)
    mae = float(overall.get("mae_bps", 0.0))
    rmse = float(overall.get("rmse_bps", 0.0))
    bias = float(overall.get("bias_bps", 0.0))

    tmpl = Template("""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>$title</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu;max-width:900px;margin:2rem auto;padding:0 1rem;}
 h1{font-size:1.6rem;margin-bottom:.25rem}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}
 .card{border:1px solid #e6e6e6;border-radius:10px;padding:14px}
 .kpis{display:flex;gap:16px;flex-wrap:wrap}
 .kpi{padding:10px 12px;border:1px solid #eee;border-radius:8px;background:#fafafa}
 img{max-width:100%;height:auto;border:1px solid #eee;border-radius:8px}
 code{background:#f4f4f4;padding:2px 4px;border-radius:4px}
 footer{margin-top:2rem;color:#666;font-size:.9rem}
</style>
</head>
<body>
  <h1>$title</h1>
  <p><strong>Folder:</strong> <code>$folder</code></p>

  <div class="kpis">
    <div class="kpi"><strong>n</strong><br>$n</div>
    <div class="kpi"><strong>MAE (bps)</strong><br>$mae</div>
    <div class="kpi"><strong>RMSE (bps)</strong><br>$rmse</div>
    <div class="kpi"><strong>Bias (bps)</strong><br>$bias</div>
  </div>

  <div class="grid" style="margin-top:1rem">
    <div class="card">
      <h3>Calibration</h3>
      <img src="figures/calibration.png" alt="Calibration plot">
    </div>
    <div class="card">
      <h3>Residuals</h3>
      <img src="figures/residual_hist.png" alt="Residual histogram">
    </div>
  </div>

  <details style="margin-top:1rem">
    <summary>Raw metrics.json</summary>
    <pre>$metrics_json</pre>
  </details>

  <footer>Generated by pt-liqadj</footer>
</body>
</html>
""")

    html = tmpl.substitute(
        title=title,
        folder=str(backtest_dir),
        n=str(n),
        mae=f"{mae:.3f}",
        rmse=f"{rmse:.3f}",
        bias=f"{bias:.3f}",
        metrics_json=json.dumps(metrics, indent=2),
    )
    out = backtest_dir / "report.html"
    out.write_text(html, encoding="utf-8")
    return out
