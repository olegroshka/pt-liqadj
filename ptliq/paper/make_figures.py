# ptliq/paper/make_figures.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless safe
import matplotlib.pyplot as plt


#FORMATS = ("png", "pdf")
FORMATS = ("png")


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
    tables_dir = Path(tables_dir)
    # tolerate common typos / legacy names
    warm = _read_csv_safe(tables_dir / "warm_scenarios.csv")
    if warm is None:
        warm = _read_csv_safe(tables_dir / "warm_scenarious.csv")
    cold = _read_csv_safe(tables_dir / "cold_scenarios.csv")
    if cold is None:
        cold = _read_csv_safe(tables_dir / "cold_scenarious.csv")
    drift = _read_csv_safe(tables_dir / "portfolio_drift.csv")
    abla  = _read_csv_safe(tables_dir / "ablation.csv")
    par   = _read_csv_safe(tables_dir / "parity.csv")
    test_pf = _read_csv_safe(tables_dir / "test_portfolios_eval.csv")
    single_line = _read_csv_safe(tables_dir / "single_line_eval.csv")

    # normalize column names used below
    def _norm(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if df is None: return None
        c = {k: k.strip().lower() for k in df.columns}
        df = df.rename(columns=c)
        return df
    return TableBundle(
        warm=_norm(warm),
        cold=_norm(cold),
        drift=_norm(drift),
        ablation=_norm(abla),
        parity=_norm(par),
        test_pf=_norm(test_pf),
        single_line=_norm(single_line),
    )


# -----------------------------
# Helpers
# -----------------------------
def _save_figure(fig: plt.Figure, outdir: Path, stem: str, fmts: Tuple[str, ...]) -> List[str]:
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in fmts:
        path = outdir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        paths.append(str(path))
    plt.close(fig)
    return paths


def _scenario_pivot(df: pd.DataFrame, value_col: str = "pred_scorer_bps") -> pd.DataFrame:
    """
    Pivot to rows=(portfolio_id, pf_gid, isin, asof_date?), cols=scenario ∈ {A,B,C,D}.
    """
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
    # ensure all letters exist
    for s in ("A", "B", "C", "D"):
        if s not in piv.columns:
            piv[s] = np.nan
    piv = piv[["A", "B", "C", "D"]]
    return piv.sort_index()


def _delta(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _pick_numeric_col(df: pd.DataFrame, preferred: str = "pred_scorer_bps") -> str:
    if preferred in df.columns and pd.api.types.is_numeric_dtype(df[preferred]):
        return preferred
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if num_cols:
        return num_cols[-1]
    return preferred


def _warm_deltas(df_warm: pd.DataFrame) -> Dict[str, pd.Series]:
    col = _pick_numeric_col(df_warm)
    piv = _scenario_pivot(df_warm, value_col=col)
    return {
        "AB": _delta(piv["B"] - piv["A"]),
        "AC": _delta(piv["C"] - piv["A"]),
        "AD": _delta(piv["D"] - piv["A"]),
    }


def _cold_deltas(df_cold: pd.DataFrame) -> Dict[str, pd.Series]:
    col = _pick_numeric_col(df_cold)
    piv = _scenario_pivot(df_cold, value_col=col)
    return {
        "AB": _delta(piv["B"] - piv["A"]),
        "AC": _delta(piv["C"] - piv["A"]),
    }


def _histogram(series: pd.Series, title: str, xlabel: str) -> plt.Figure:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    fig = plt.figure()
    plt.hist(s.values, bins=20)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.grid(True, alpha=0.3)
    return fig


def _two_bar(values: List[Tuple[str, float]], title: str, ylabel: str) -> plt.Figure:
    labels = [k for (k, _) in values]
    vals = [float(v) for (_, v) in values]
    fig = plt.figure()
    xs = np.arange(len(vals))
    plt.bar(xs, vals)
    plt.xticks(xs, labels)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.3)
    return fig


def _named_bar(items: List[Tuple[str, float]], title: str, ylabel: str) -> plt.Figure:
    labels = [k for (k, _) in items]
    vals = [float(v) for (_, v) in items]
    fig = plt.figure()
    xs = np.arange(len(vals))
    plt.bar(xs, vals)
    plt.xticks(xs, labels, rotation=0)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.3)
    return fig


# scatter with 1:1 line and summary in subtitle
def _scatter_pred_vs_true(df: pd.DataFrame, x_col: str, y_col: str, title: str) -> plt.Figure:
    d = df[[x_col, y_col]].dropna()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(d[x_col].values, d[y_col].values, s=14, alpha=0.7)
    # 1:1 line
    vmin = float(np.nanmin([d[x_col].min(), d[y_col].min()]))
    vmax = float(np.nanmax([d[x_col].max(), d[y_col].max()]))
    lo, hi = vmin - 0.05*(vmax-vmin), vmax + 0.05*(vmax-vmin)
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_title(title)
    ax.set_xlabel("Prediction (bps)")
    ax.set_ylabel("Truth: residual (bps)")
    # correlation in subtitle
    if len(d) >= 2:
        r = float(np.corrcoef(d[x_col].values, d[y_col].values)[0,1])
        ax.text(0.02, 0.98, f"n={len(d)}, r={r:.3f}", transform=ax.transAxes,
                ha="left", va="top")
    ax.grid(True, alpha=0.3)
    return fig


# -----------------------------
# Main renderer
# -----------------------------
def _render_figs(tables: TableBundle, out: Path, fmts: Tuple[str, ...] = FORMATS) -> Dict[str, List[str]]:
    """
    Renders figures:

      F1: warm_size_elasticity (ΔB−A)
      F2: warm_side_flip      (ΔC−A)
      F3: warm_time_roll      (ΔD−A)
      F4: cold_size_elasticity (ΔB−A)
      F5: cold_side_flip       (ΔC−A)
      F6: portfolio_drift_hist (|Δ| distribution)
      F7: ablation_effect      (full vs mask0_nopf)
      F8: test_portfolios_scatter (NEW: pred vs truth on unseen portfolios)
      F9: single_line_diff_hist  (NEW: Δ with‑pf − no‑pf)
      F10: single_line_parity_scatter (NEW: pred_noctx vs pred_pf1)
    """
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, List[str]] = {}

    # --- Warm
    if tables.warm is not None:
        wd = _warm_deltas(tables.warm)
        f1 = _histogram(wd["AB"], "Warm: size elasticity (B−A)", "Δ bps")
        saved["fig_warm_size_elasticity"] = _save_figure(f1, out, "fig_warm_size_elasticity", fmts)

        f2 = _histogram(wd["AC"], "Warm: side flip (C−A)", "Δ bps")
        saved["fig_warm_side_flip"] = _save_figure(f2, out, "fig_warm_side_flip", fmts)

        f3 = _histogram(wd["AD"], "Warm: next‑day shift (D−A)", "Δ bps")
        saved["fig_warm_time_roll"] = _save_figure(f3, out, "fig_warm_time_roll", fmts)

    # --- Cold
    if tables.cold is not None:
        cd = _cold_deltas(tables.cold)
        f4 = _histogram(cd["AB"], "Cold: size elasticity (B−A)", "Δ bps")
        saved["fig_cold_size_elasticity"] = _save_figure(f4, out, "fig_cold_size_elasticity", fmts)

        f5 = _histogram(cd["AC"], "Cold: side flip (C−A)", "Δ bps")
        saved["fig_cold_side_flip"] = _save_figure(f5, out, "fig_cold_side_flip", fmts)

    # --- Portfolio drift
    if tables.drift is not None:
        col = "abs_delta_bps" if "abs_delta_bps" in tables.drift.columns else list(tables.drift.columns)[-1]
        f6 = _histogram(tables.drift[col], "Portfolio drift: |Δ| across anchors", "|Δ| bps")
        saved["fig_portfolio_drift_hist"] = _save_figure(f6, out, "fig_portfolio_drift_hist", fmts)

    # --- Ablation
    if tables.ablation is not None:
        full = float(tables.ablation.get("delta_full_bps", pd.Series([np.nan])).iloc[0])
        mask = float(tables.ablation.get("delta_mask0_nopf_bps", pd.Series([np.nan])).iloc[0])
        f7 = _two_bar([("full", full), ("mask0_nopf", mask)], "Ablation: Δ (B−A) under settings", "Δ bps")
        saved["fig_ablation"] = _save_figure(f7, out, "fig_ablation", fmts)

    # --- Unseen portfolio scatter (prediction vs truth)
    if tables.test_pf is not None:
        df = tables.test_pf.copy()
        # prefer pred_bps, residual_bps columns (created by the new experiment.py)
        if ("pred_bps" in df.columns) and ("residual_bps" in df.columns):
            f8 = _scatter_pred_vs_true(df, x_col="pred_bps", y_col="residual_bps",
                                       title="Unseen portfolios: prediction vs truth (bps)")
            saved["fig_test_pf_scatter"] = _save_figure(f8, out, "fig_test_pf_scatter", fmts)

    # --- Single-line pseudo-portfolio vs baseline
    if tables.single_line is not None and len(tables.single_line) > 0:
        s = tables.single_line.copy()
        if {"pred_pf1_bps","pred_noctx_bps"}.issubset(s.columns):
            # diff histogram
            f9 = _histogram(s["diff_bps"], "Single-line portfolios behave like baseline trades", "Δ (with‑pf − no‑pf) bps")
            saved["fig_single_line_diff_hist"] = _save_figure(f9, out, "fig_single_line_diff_hist", fmts)
            # parity scatter
            f10 = _scatter_pred_vs_true(s.rename(columns={"pred_noctx_bps":"x", "pred_pf1_bps":"y"}),
                                        x_col="x", y_col="y",
                                        title="Single-line parity: baseline vs 1‑line portfolio (bps)")
            saved["fig_single_line_parity_scatter"] = _save_figure(f10, out, "fig_single_line_parity_scatter", fmts)

    return saved


# Public alias
def render_figs(tables_dir: Path, out: Path, fmts: Tuple[str, ...] = FORMATS) -> Dict[str, List[str]]:
    bundle = load_tables(Path(tables_dir))
    return _render_figs(bundle, Path(out), fmts)
