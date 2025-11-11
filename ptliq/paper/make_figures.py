# ptliq/paper/make_figures.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # safe headless
import matplotlib.pyplot as plt
import seaborn as sns  # new

# ---- Formats: PNG-only by default (fix the tuple bug) ----
FORMATS: Tuple[str, ...] = ("png",)

# ---- Seaborn theme for paper figures ----
def _set_theme() -> None:
    # A compact, readable style for papers
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.2)

# -----------------------------
# Data loading
# -----------------------------
@dataclass
class TableBundle:
    warm: Optional[pd.DataFrame]
    cold: Optional[pd.DataFrame]
    drift: Optional[pd.DataFrame]
    ablation: Optional[pd.DataFrame]
    parity: Optional[pd.DataFrame]
    test_pf: Optional[pd.DataFrame]
    single_line: Optional[pd.DataFrame]

def _read_csv_safe(p: Path) -> Optional[pd.DataFrame]:
    try:
        if p.exists():
            return pd.read_csv(p)
    except Exception:
        pass
    return None

def load_tables(tables_dir: Path) -> TableBundle:
    t = Path(tables_dir)

    # tolerate legacy names without triggering DataFrame truthiness
    warm = _read_csv_safe(t / "warm_scenarios.csv")
    if warm is None:
        warm = _read_csv_safe(t / "warm_scenarious.csv")
    cold = _read_csv_safe(t / "cold_scenarios.csv")
    if cold is None:
        cold = _read_csv_safe(t / "cold_scenarious.csv")
    drift = _read_csv_safe(t / "portfolio_drift.csv")
    abla  = _read_csv_safe(t / "ablation.csv")
    par   = _read_csv_safe(t / "parity.csv")
    test_pf = _read_csv_safe(t / "test_portfolios_eval.csv")
    single_line = _read_csv_safe(t / "single_line_eval.csv")

    def _norm(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if df is None: return None
        return df.rename(columns={k: k.strip().lower() for k in df.columns})

    return TableBundle(
        warm=_norm(warm), cold=_norm(cold), drift=_norm(drift),
        ablation=_norm(abla), parity=_norm(par),
        test_pf=_norm(test_pf), single_line=_norm(single_line)
    )

# -----------------------------
# Helpers
# -----------------------------
def _save_figure(fig: plt.Figure, outdir: Path, stem: str, fmts: Tuple[str, ...]) -> List[str]:
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []
    for ext in fmts:
        path = outdir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        paths.append(str(path))
    plt.close(fig)
    return paths

def _scenario_pivot(df: pd.DataFrame, value_col: str = "pred_scorer_bps") -> pd.DataFrame:
    keys = [k for k in ("portfolio_id", "pf_gid", "isin", "asof_date") if k in df.columns]
    if not keys:
        raise ValueError("scenario CSV missing expected identifier columns")
    if "scenario" not in df.columns:
        raise ValueError("scenario CSV missing 'scenario' column")
    if value_col not in df.columns:
        nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not nums:
            raise ValueError("no numeric prediction column found")
        value_col = nums[-1]
    piv = df.pivot_table(index=keys, columns="scenario", values=value_col, aggfunc="first")
    for s in ("A","B","C","D"):
        if s not in piv.columns:
            piv[s] = np.nan
    return piv[["A","B","C","D"]].sort_index()

def _delta(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def _pick_numeric_col(df: pd.DataFrame, preferred: str = "pred_scorer_bps") -> str:
    if preferred in df.columns and pd.api.types.is_numeric_dtype(df[preferred]):
        return preferred
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return nums[-1] if nums else preferred

def _warm_deltas(df: pd.DataFrame) -> Dict[str, pd.Series]:
    col = _pick_numeric_col(df); piv = _scenario_pivot(df, col)
    return {"AB": _delta(piv["B"] - piv["A"]),
            "AC": _delta(piv["C"] - piv["A"]),
            "AD": _delta(piv["D"] - piv["A"])}

def _cold_deltas(df: pd.DataFrame) -> Dict[str, pd.Series]:
    col = _pick_numeric_col(df); piv = _scenario_pivot(df, col)
    return {"AB": _delta(piv["B"] - piv["A"]),
            "AC": _delta(piv["C"] - piv["A"])}

# ——— Vis helpers (seaborn) ———
def _hist_with_kde(x: pd.Series, title: str, xlabel: str, color: str = "C0") -> plt.Figure:
    s = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    if len(s) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    else:
        sns.histplot(s, bins=24, kde=True, ax=ax, color=color, edgecolor="white")
        med = float(s.median()); iqr = float(s.quantile(0.75) - s.quantile(0.25))
        ax.axvline(med, color="black", linestyle="--", linewidth=1.0, label=f"median = {med:.2f} bps")
        ax.set_title(title)
        ax.set_xlabel(xlabel); ax.set_ylabel("Count")
        ax.legend(loc="best", frameon=True)
        ax.text(0.02, 0.98, f"n={len(s)}, IQR={iqr:.2f} bps", transform=ax.transAxes,
                ha="left", va="top")
    return fig

def _two_bar(items: List[Tuple[str, float]], title: str, ylabel: str) -> plt.Figure:
    labels = [k for k,_ in items]; vals = [float(v) for _,v in items]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    sns.barplot(x=labels, y=vals, ax=ax)
    ax.set_title(title); ax.set_ylabel(ylabel); ax.set_xlabel("")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    return fig

def _scatter_pred_vs_true(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    hue: Optional[str] = None,
    hue_norm: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    d = df[[x_col, y_col] + ([hue] if hue and hue in df.columns else [])].dropna()
    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    # scatter + (optional) hue
    if hue and hue in d.columns:
        sns.scatterplot(data=d, x=x_col, y=y_col, hue=hue, ax=ax, s=30, alpha=0.7, edgecolor="none")
        ax.legend(title=hue, loc="best", frameon=True)
    else:
        sns.scatterplot(data=d, x=x_col, y=y_col, ax=ax, s=30, alpha=0.7, edgecolor="none")
    # regression (thin) and 45° line
    if len(d) >= 3:
        sns.regplot(data=d, x=x_col, y=y_col, ax=ax, scatter=False, ci=None, line_kws=dict(lw=1.0, ls="-", color="C1"))
    lims = [min(d[x_col].min(), d[y_col].min()), max(d[x_col].max(), d[y_col].max())]
    pad = 0.05 * (lims[1] - lims[0] if lims[1] > lims[0] else 1.0)
    lo, hi = lims[0] - pad, lims[1] + pad
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.0, color="gray", label="45° line")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    # metrics
    if len(d) >= 2:
        x = d[x_col].to_numpy(); y = d[y_col].to_numpy()
        r = float(np.corrcoef(x, y)[0,1])
        mae = float(np.mean(np.abs(x - y)))
        rmse = float(np.sqrt(np.mean((x - y) ** 2)))
        ax.text(0.02, 0.98, f"n={len(d)}, r={r:.3f}, MAE={mae:.2f}, RMSE={rmse:.2f}",
                transform=ax.transAxes, ha="left", va="top")
    ax.set_title(title)
    ax.set_xlabel("Prediction (bps)")
    ax.set_ylabel("Truth: residual (bps)")
    return fig

# -----------------------------
# Main renderer
# -----------------------------
def _render_figs(tables: TableBundle, out: Path, fmts: Tuple[str, ...] = FORMATS) -> Dict[str, List[str]]:
    _set_theme()
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, List[str]] = {}

    # ---- Warm
    if tables.warm is not None and len(tables.warm) > 0:
        wd = _warm_deltas(tables.warm)
        saved["fig_warm_size_elasticity"] = _save_figure(
            _hist_with_kde(wd["AB"], "Warm: size elasticity (B–A)", "Δ (bps)", color="C0"),
            out, "fig_warm_size_elasticity", fmts)
        saved["fig_warm_side_flip"] = _save_figure(
            _hist_with_kde(wd["AC"], "Warm: side flip (C–A)", "Δ (bps)", color="C2"),
            out, "fig_warm_side_flip", fmts)
        # Only render D–A if any data present
        if pd.to_numeric(wd["AD"], errors="coerce").dropna().size > 0:
            saved["fig_warm_time_roll"] = _save_figure(
                _hist_with_kde(wd["AD"], "Warm: next‑day shift (D–A)", "Δ (bps)", color="C1"),
                out, "fig_warm_time_roll", fmts)

    # ---- Cold
    if tables.cold is not None and len(tables.cold) > 0:
        cd = _cold_deltas(tables.cold)
        saved["fig_cold_size_elasticity"] = _save_figure(
            _hist_with_kde(cd["AB"], "Cold: size elasticity (B–A)", "Δ (bps)", color="C0"),
            out, "fig_cold_size_elasticity", fmts)
        saved["fig_cold_side_flip"] = _save_figure(
            _hist_with_kde(cd["AC"], "Cold: side flip (C–A)", "Δ (bps)", color="C2"),
            out, "fig_cold_side_flip", fmts)

    # ---- Portfolio drift
    if tables.drift is not None and len(tables.drift) > 0:
        col = "abs_delta_bps" if "abs_delta_bps" in tables.drift.columns else tables.drift.select_dtypes(include="number").columns[-1]
        s = pd.to_numeric(tables.drift[col], errors="coerce").dropna()
        title = f"Portfolio drift: |Δ| across anchors (mean={s.mean():.2f} bps, median={s.median():.2f} bps)" if len(s) else "Portfolio drift: |Δ| across anchors"
        saved["fig_portfolio_drift_hist"] = _save_figure(
            _hist_with_kde(s, title, "|Δ| (bps)", color="C3"),
            out, "fig_portfolio_drift_hist", fmts)

    # ---- Ablation
    if tables.ablation is not None and len(tables.ablation) > 0:
        full = float(tables.ablation.get("delta_full_bps", pd.Series([np.nan])).iloc[0])
        mask = float(tables.ablation.get("delta_mask0_nopf_bps", pd.Series([np.nan])).iloc[0])
        saved["fig_ablation"] = _save_figure(
            _two_bar([("full", full), ("mask0_nopf", mask)], "Ablation: Δ (B–A) under settings", "Δ (bps)"),
            out, "fig_ablation", fmts)

    # ---- Unseen portfolios: pred vs truth
    if tables.test_pf is not None and len(tables.test_pf) > 0:
        df = tables.test_pf.copy()
        # Optional hue: portfolio size if available
        if "portfolio_id" in df.columns:
            pf_len = df.groupby("portfolio_id")["portfolio_id"].transform("count")
            df["pf_lines"] = pf_len
            hue = "pf_lines"
        else:
            hue = None
        if {"pred_bps", "residual_bps"}.issubset(df.columns):
            saved["fig_test_pf_scatter"] = _save_figure(
                _scatter_pred_vs_true(df, "pred_bps", "residual_bps",
                                      "Unseen portfolios: prediction vs truth (bps)",
                                      hue=hue),
                out, "fig_test_pf_scatter", fmts)

    # ---- Single-line behavior
    if tables.single_line is not None and len(tables.single_line) > 0:
        s = tables.single_line.copy()
        # Difference hist
        if "diff_bps" in s.columns:
            saved["fig_single_line_diff_hist"] = _save_figure(
                _hist_with_kde(s["diff_bps"], "Single‑line portfolios behave like baseline trades", "Δ (with‑pf – no‑pf) (bps)", color="C4"),
                out, "fig_single_line_diff_hist", fmts)
        # Parity scatter
        if {"pred_noctx_bps", "pred_pf1_bps"}.issubset(s.columns):
            s2 = s.rename(columns={"pred_noctx_bps": "pred_baseline", "pred_pf1_bps": "pred_pf1"})
            saved["fig_single_line_parity_scatter"] = _save_figure(
                _scatter_pred_vs_true(s2, "pred_baseline", "pred_pf1",
                                      "Single‑line parity: baseline vs 1‑line portfolio (bps)"),
                out, "fig_single_line_parity_scatter", fmts)

    return saved

# Public alias
def render_figs(tables_dir: Path, out: Path, fmts: Tuple[str, ...] = FORMATS) -> Dict[str, List[str]]:
    bundle = load_tables(Path(tables_dir))
    return _render_figs(bundle, Path(out), fmts)
