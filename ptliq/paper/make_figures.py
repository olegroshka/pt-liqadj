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


# -----------------------------
# Data loading
# -----------------------------
@dataclass
class TableBundle:
    warm: Optional[pd.DataFrame]
    cold: Optional[pd.DataFrame]
    drift: Optional[pd.DataFrame]
    ablation: Optional[pd.DataFrame]
    neg_drag: Optional[pd.DataFrame]
    parity: Optional[pd.DataFrame]


def _read_csv_safe(p: Path) -> Optional[pd.DataFrame]:
    try:
        if p.exists():
            return pd.read_csv(p)
    except Exception:
        pass
    return None


def load_tables(tables_dir: Path) -> TableBundle:
    tables_dir = Path(tables_dir)
    # tolerate common typos / legacy names — without DataFrame boolean evaluation
    warm = _read_csv_safe(tables_dir / "warm_scenarios.csv")
    if warm is None:
        warm = _read_csv_safe(tables_dir / "warm_scenarious.csv")
    cold = _read_csv_safe(tables_dir / "cold_scenarios.csv")
    if cold is None:
        cold = _read_csv_safe(tables_dir / "cold_scenarious.csv")
    drift = _read_csv_safe(tables_dir / "portfolio_drift.csv")
    abla  = _read_csv_safe(tables_dir / "ablation.csv")
    nd    = _read_csv_safe(tables_dir / "negative_drag.csv")
    par   = _read_csv_safe(tables_dir / "parity.csv")
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
        neg_drag=_norm(nd),
        parity=_norm(par),
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
    Any missing scenario becomes NaN.
    """
    # establish a stable index that exists in both warm/cold
    keys = [k for k in ("portfolio_id", "pf_gid", "isin", "asof_date") if k in df.columns]
    if not keys:
        raise ValueError("scenario CSV missing expected identifier columns")
    if "scenario" not in df.columns:
        raise ValueError("scenario CSV missing 'scenario' column")
    if value_col not in df.columns:
        # fallback to last numeric col
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
    # Prefer a known column if it exists and is numeric
    if preferred in df.columns and pd.api.types.is_numeric_dtype(df[preferred]):
        return preferred
    # Otherwise choose the last numeric column if any
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if num_cols:
        return num_cols[-1]
    # Fallback to preferred even if non-numeric; _delta will coerce later after pivot subtraction
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
    # cold has A,B,C only
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


# -----------------------------
# Main renderer (the function you were missing)
# -----------------------------
def _render_figs(tables: TableBundle, out: Path, fmts: Tuple[str, ...] = ("png", "pdf")) -> Dict[str, List[str]]:
    """
    Renders eight figures from the CSV bundle:

      F1: warm_size_elasticity (ΔB−A)
      F2: warm_side_flip      (ΔC−A)
      F3: warm_time_roll      (ΔD−A)
      F4: cold_size_elasticity (ΔB−A)
      F5: cold_side_flip       (ΔC−A)
      F6: portfolio_drift_hist (|Δ| distribution)
      F7: ablation_effect      (full vs mask0_nopf)
      F8: negative_drag        (BUY/SELL deltas)

    Returns a dict: {figure_stem: [saved_paths...], ...}
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
        # column normalization
        cols = {c: c for c in tables.ablation.columns}
        for k in list(cols):
            if k.startswith("delta_"): continue
        full = float(tables.ablation.get("delta_full_bps", pd.Series([np.nan])).iloc[0])
        mask = float(tables.ablation.get("delta_mask0_nopf_bps", pd.Series([np.nan])).iloc[0])
        f7 = _two_bar([("full", full), ("mask0_nopf", mask)], "Ablation: Δ (B−A) under settings", "Δ bps")
        saved["fig_ablation"] = _save_figure(f7, out, "fig_ablation", fmts)

    # --- Negative drag
    if tables.neg_drag is not None:
        # Expect columns: side, delta_bps
        df = tables.neg_drag.copy()
        if "delta_bps" not in df.columns:
            # fallback to last numeric
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if not num_cols:
                raise ValueError("negative_drag.csv has no numeric column")
            df["delta_bps"] = df[num_cols[-1]]
        rows: List[Tuple[str, float]] = []
        for side in ["BUY", "SELL"]:
            m = df[df["side"].str.upper() == side] if "side" in df.columns and df["side"].dtype == object else df
            if len(m) > 0:
                rows.append((side, float(m["delta_bps"].mean())))
        f8 = _named_bar(rows, "Negative drag (expected ≤ 0)", "Δ bps = y_drag − y_anchor")
        saved["fig_negative_drag"] = _save_figure(f8, out, "fig_negative_drag", fmts)

    return saved


# Public alias (so external callers don’t need the underscore)
def render_figs(tables_dir: Path, out: Path, fmts: Tuple[str, ...] = ("png", "pdf")) -> Dict[str, List[str]]:
    bundle = load_tables(Path(tables_dir))
    return _render_figs(bundle, Path(out), fmts)
