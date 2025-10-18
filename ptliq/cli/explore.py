from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence

import math
import pandas as pd
import numpy as np
import typer
from rich import print
from rich.table import Table
from rich.console import Console

app = typer.Typer(no_args_is_help=True)


def _load_parquet(path: Path, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    Load a parquet file optionally selecting a subset of columns.
    If requested columns are missing in the file, do not fail —
    load what exists and warn about the missing ones.
    """
    if not columns:
        return pd.read_parquet(path)
    try:
        return pd.read_parquet(path, columns=list(columns))
    except KeyError:
        # Some requested columns are not present in the file. Load all columns,
        # then filter down to the intersection and warn the user.
        df = pd.read_parquet(path)
        req = list(columns)
        existing = [c for c in req if c in df.columns]
        missing = [c for c in req if c not in df.columns]
        if missing:
            print("[yellow]Warning:[/yellow] skipping missing columns → " + ", ".join(missing))
        if existing:
            return df[existing]
        # If none of the requested columns exist, fall back to full DataFrame
        # so the user can still explore the file.
        return df


def _is_categorical(series: pd.Series, max_unique_ratio: float = 0.2, max_uniques: int = 50) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return True
    if pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
        # numeric with very few uniques can be treated as categorical
        n = len(series)
        uniq = series.nunique(dropna=True)
        return uniq <= max_uniques and (n > 0 and uniq / n <= max_unique_ratio)
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series) or pd.api.types.is_string_dtype(series) or pd.api.types.is_datetime64_any_dtype(series)


def _print_schema(df: pd.DataFrame, console: Console) -> None:
    t = Table(title="Schema", show_lines=False)
    t.add_column("column")
    t.add_column("dtype")
    t.add_column("non_null", justify="right")
    t.add_column("nulls", justify="right")
    t.add_column("unique", justify="right")
    for col in df.columns:
        s = df[col]
        t.add_row(col, str(s.dtype), f"{s.notna().sum():,}", f"{s.isna().sum():,}", f"{s.nunique(dropna=True):,}")
    console.print(t)


def _print_quick_stats(df: pd.DataFrame, console: Console) -> None:
    if df.empty:
        console.print("[yellow]No rows[/yellow]")
        return
    # numeric stats
    num_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
    ]
    if num_cols:
        t = Table(title="Numeric stats", show_lines=False)
        for c in ["column", "count", "mean", "std", "min", "p50", "max"]:
            t.add_column(c, justify="right" if c != "column" else "left")
        desc = df[num_cols].describe(percentiles=[0.5]).T
        for col in num_cols:
            row = desc.loc[col]
            t.add_row(
                col,
                f"{int(row['count']):,}",
                f"{row['mean']:.4g}" if not math.isnan(row['mean']) else "-",
                f"{row['std']:.4g}" if not math.isnan(row['std']) else "-",
                f"{row['min']:.4g}",
                f"{row['50%']:.4g}",
                f"{row['max']:.4g}",
            )
        console.print(t)

    # non-numeric quick info: show top-5 for each
    other_cols = [c for c in df.columns if c not in num_cols]
    if other_cols:
        for col in other_cols:
            s = df[col]
            vc = s.value_counts(dropna=True).head(5)
            t = Table(title=f"Top values: {col}")
            t.add_column("value")
            t.add_column("count", justify="right")
            for idx, cnt in vc.items():
                t.add_row(str(idx), f"{int(cnt):,}")
            console.print(t)


def _histogram(series: pd.Series, bins: int = 20):
    s = series.dropna()
    if s.empty:
        return None
    if pd.api.types.is_datetime64_any_dtype(s):
        # convert to int for binning (nanoseconds) using astype (view is deprecated)
        s = s.astype('int64')
    if pd.api.types.is_bool_dtype(s):
        # Bin booleans explicitly to avoid bool→uint8 histogram warnings
        counts, edges = np.histogram(s.astype(int), bins=[-0.5, 0.5, 1.5])
        return counts, np.array([-0.5, 0.5, 1.5])
    counts, edges = np.histogram(s, bins=bins)
    return counts, edges


def _print_distributions(df: pd.DataFrame, console: Console, topk: int, bins: int) -> None:
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
            h = _histogram(s, bins=bins)
            if h is None:
                console.print(f"[yellow]{col}[/yellow]: empty")
                continue
            counts, edges = h
            t = Table(title=f"Histogram: {col}")
            t.add_column("bin")
            t.add_column("range")
            t.add_column("count", justify="right")
            for i in range(len(counts)):
                lo = edges[i]
                hi = edges[i + 1]
                # pretty print numbers or datetimes
                if pd.api.types.is_datetime64_any_dtype(s):
                    lo_p = pd.to_datetime(int(lo))
                    hi_p = pd.to_datetime(int(hi))
                else:
                    lo_p = lo
                    hi_p = hi
                t.add_row(str(i + 1), f"[{lo_p} .. {hi_p})", f"{int(counts[i]):,}")
            console.print(t)
        else:
            vc = s.astype(str).replace("<NA>", np.nan).dropna().value_counts().head(topk)
            t = Table(title=f"Frequencies: {col}")
            t.add_column("value")
            t.add_column("count", justify="right")
            for v, c in vc.items():
                t.add_row(str(v), f"{int(c):,}")
            console.print(t)


def _print_correlations(df: pd.DataFrame, console: Console, method: str = "pearson", top_pairs: int = 10) -> None:
    num_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
    ]
    if len(num_cols) < 2:
        console.print("[yellow]Not enough numeric columns for correlations[/yellow]")
        return
    corr = df[num_cols].corr(method=method)
    # Print top correlated pairs by absolute value (excluding self)
    pairs = []
    for i, a in enumerate(num_cols):
        for j, b in enumerate(num_cols):
            if j <= i:
                continue
            val = corr.loc[a, b]
            if pd.isna(val):
                continue
            pairs.append((a, b, float(val), abs(float(val))))
    pairs.sort(key=lambda x: x[3], reverse=True)
    t = Table(title=f"Top {min(top_pairs, len(pairs))} correlations ({method})", show_lines=False)
    t.add_column("col_A")
    t.add_column("col_B")
    t.add_column("corr", justify="right")
    for a, b, v, _ in pairs[:top_pairs]:
        t.add_row(a, b, f"{v:.3f}")
    console.print(t)


def _save_plots(
    df: pd.DataFrame,
    outdir: Path,
    bins: int = 20,
    max_hists: int = 10,
    plot_prefix: Optional[str] = None,
    show: bool = True,
) -> list[Path]:
    written: list[Path] = []
    try:
        import matplotlib.pyplot as plt  # type: ignore
        try:
            import seaborn as sns  # type: ignore
        except Exception:
            sns = None  # type: ignore
        from matplotlib import gridspec  # type: ignore
        import math as _m
    except Exception as e:
        print("[yellow]matplotlib not installed; skipping plots. Install with: pip install matplotlib seaborn[/yellow]")
        return written

    outdir.mkdir(parents=True, exist_ok=True)

    # Identify columns
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    bool_cols = [c for c in df.columns if pd.api.types.is_bool_dtype(df[c])]

    # 1) Save correlation heatmap
    heat_path = None
    if len(num_cols) >= 2:
        corr = df[num_cols].corr(numeric_only=True)
        fig_h = plt.figure(figsize=(max(6, len(num_cols) * 0.6), max(5, len(num_cols) * 0.6)))
        ax_h = fig_h.add_subplot(111)
        if 'sns' in locals() and sns is not None:
            sns.heatmap(corr, annot=False, cmap="vlag", center=0.0, ax=ax_h)
        else:
            im = ax_h.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
            fig_h.colorbar(im, ax=ax_h)
            ax_h.set_xticks(range(len(num_cols)))
            ax_h.set_yticks(range(len(num_cols)))
            ax_h.set_xticklabels(num_cols, rotation=90)
            ax_h.set_yticklabels(num_cols)
        ax_h.set_title("Correlation heatmap")
        heat_name = (f"{plot_prefix}__correlation_heatmap.png" if plot_prefix else "correlation_heatmap.png")
        heat_path = outdir / heat_name
        fig_h.tight_layout()
        fig_h.savefig(heat_path, dpi=200)
        plt.close(fig_h)
        written.append(heat_path)

    # 2) Individual numeric histograms
    hist_series = []  # collect for composite figure
    for c in num_cols[:max_hists]:
        s = df[c].dropna()
        if s.empty:
            continue
        fig_i = plt.figure(figsize=(5, 3))
        ax_i = fig_i.add_subplot(111)
        ax_i.hist(s.values.astype(float), bins=bins, color="#4C78A8", edgecolor="white")
        ax_i.set_title(f"Histogram: {c}")
        ax_i.set_xlabel(c)
        ax_i.set_ylabel("count")
        hist_name = (f"{plot_prefix}__hist_{c}.png" if plot_prefix else f"hist_{c}.png")
        path = outdir / hist_name
        fig_i.tight_layout()
        fig_i.savefig(path, dpi=160)
        plt.close(fig_i)
        written.append(path)
        hist_series.append((c, s))

    # 3) Overview figure: heatmap + auto-selected histograms (numeric + datetime + bool)
    to_plot_cols = num_cols[:6] + dt_cols[:2] + bool_cols[:2]
    if len(to_plot_cols) > 0:
        # rows: 1 for heatmap + ceil(n_hist/3)
        n_hist = len(to_plot_cols)
        rows_hists = _m.ceil(n_hist / 3)
        rows_total = 1 + rows_hists
        fig = plt.figure(figsize=(10, 3.2 * rows_total))
        gs = fig.add_gridspec(nrows=rows_total, ncols=3)

        # Heatmap on top
        ax0 = fig.add_subplot(gs[0, :])
        if len(num_cols) >= 2:
            corr2 = df[num_cols].corr(numeric_only=True)
            if 'sns' in locals() and sns is not None:
                sns.heatmap(corr2, ax=ax0, cmap="coolwarm", center=0, cbar=True)
            else:
                im = ax0.imshow(corr2.values, cmap="coolwarm", vmin=-1, vmax=1)
                fig.colorbar(im, ax=ax0)
                ax0.set_xticks(range(len(corr2.columns)))
                ax0.set_xticklabels(corr2.columns, rotation=90, fontsize=8)
                ax0.set_yticks(range(len(corr2.columns)))
                ax0.set_yticklabels(corr2.columns, fontsize=8)
            ax0.set_title("Correlation heatmap")
        else:
            ax0.axis("off")
            ax0.set_title("Correlation heatmap (insufficient numeric cols)")

        # Histograms
        cell_idx = 0
        for cname in to_plot_cols:
            r = 1 + (cell_idx // 3)
            cidx = cell_idx % 3
            ax = fig.add_subplot(gs[r, cidx])
            s = df[cname].dropna()
            if pd.api.types.is_datetime64_any_dtype(s):
                s_i = s.astype("int64")
                counts, edges = np.histogram(s_i, bins=bins)
                ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge")
                ax.set_xlabel(cname)
                ax.set_ylabel("count")
                ax.set_title(f"Histogram: {cname}")
            elif pd.api.types.is_bool_dtype(s):
                counts, edges = np.histogram(s.astype(int), bins=[-0.5, 0.5, 1.5])
                ax.bar([0, 1], counts, align="center")
                ax.set_xticks([0, 1])
                ax.set_xticklabels(["False", "True"])
                ax.set_ylabel("count")
                ax.set_title(f"Histogram: {cname}")
            elif pd.api.types.is_numeric_dtype(s):
                ax.hist(s.values.astype(float), bins=bins)
                ax.set_xlabel(cname)
                ax.set_ylabel("count")
                ax.set_title(f"Histogram: {cname}")
            else:
                vc = s.astype(str).value_counts().head(20)
                ax.bar(vc.index, vc.values)
                ax.tick_params(axis="x", rotation=90, labelsize=7)
                ax.set_title(f"Freq: {cname}")
            cell_idx += 1

        fig.tight_layout()
        overview_name = (f"{plot_prefix}__overview.png" if plot_prefix else "overview.png")
        overview_path = outdir / overview_name
        fig.savefig(overview_path, dpi=200)
        plt.close(fig)
        written.append(overview_path)

    # 4) Optional consolidated window
    if show:
        try:
            n_h = len(hist_series)
            has_heat = heat_path is not None
            cols = 3 if n_h >= 3 else max(1, n_h)
            rows_hists = (n_h + cols - 1) // cols if n_h > 0 else 0
            rows_total = rows_hists + (1 if has_heat else 0)
            fig_height = 3.0 * rows_total + (2.0 if has_heat else 0.0)
            fig = plt.figure(figsize=(max(9, cols * 4), max(6, fig_height)))
            gs = gridspec.GridSpec(rows_total, cols, height_ratios=[2.0] + [1.0] * rows_hists if has_heat else [1.0] * rows_hists)

            idx_row = 0
            if has_heat:
                ax = fig.add_subplot(gs[0, :])
                corr2 = df[num_cols].corr() if len(num_cols) >= 2 else None
                if corr2 is not None:
                    if 'sns' in locals() and sns is not None:
                        sns.heatmap(corr2, annot=False, cmap="vlag", center=0.0, ax=ax)
                    else:
                        im = ax.imshow(corr2, cmap="RdBu_r", vmin=-1, vmax=1)
                        fig.colorbar(im, ax=ax)
                        ax.set_xticks(range(len(num_cols)))
                        ax.set_yticks(range(len(num_cols)))
                        ax.set_xticklabels(num_cols, rotation=90)
                        ax.set_yticklabels(num_cols)
                    ax.set_title("Correlation heatmap")
                idx_row = 1

            for i, (cname, sdata) in enumerate(hist_series):
                r = idx_row + (i // cols)
                col = i % cols
                ax = fig.add_subplot(gs[r, col])
                ax.hist(sdata, bins=bins, color="#4C78A8", edgecolor="white")
                ax.set_title(f"Hist: {cname}")
                ax.set_xlabel(cname)
                ax.set_ylabel("count")

            fig.tight_layout()
            plt.show()
        except Exception:
            print("[yellow]Could not open plot window (headless environment?). Files were saved to disk.[/yellow]")
        finally:
            try:
                plt.close('all')
            except Exception:
                pass

    return written


def _write_pdf(df: pd.DataFrame, outdir: Path, stem: str, bins: int = 20) -> Path | None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib.backends.backend_pdf import PdfPages  # type: ignore
        try:
            import seaborn as sns  # type: ignore
        except Exception:
            sns = None  # type: ignore
    except Exception:
        print("[yellow]matplotlib not installed; skipping --pdf. Install with: pip install matplotlib seaborn[/yellow]")
        return None

    outdir.mkdir(parents=True, exist_ok=True)
    pdf_path = outdir / f"{stem}__report.pdf"

    # Helper to add a text page
    def add_text_page(title: str, lines: list[str]):
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        txt = [f"{title}", ""] + lines
        ax.text(0.02, 0.98, "\n".join(txt), va="top", family="monospace", fontsize=9)
        pdf_out.savefig(fig)
        plt.close(fig)

    # Column groups
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    bool_cols = [c for c in df.columns if pd.api.types.is_bool_dtype(df[c])]

    with PdfPages(pdf_path) as pdf_out:
        # Cover / overview page with guidance
        lines = [
            f"Rows: {len(df):,} | Columns: {df.shape[1]}",
            "",
            "This report summarizes the dataset with schema, stats, correlations, and distributions.",
            "Use it to spot data issues:",
            "- Missing data: high null counts, empty histograms, or spikes at special values.",
            "- Outliers: very wide ranges or long tails in histograms.",
            "- Duplicates / low cardinality: tiny 'unique' vs rows for categorical columns.",
            "- Leakage / redundancy: very high correlations (|r| ≈ 1).",
        ]
        add_text_page("Data exploration report", lines)

        # Schema page with description
        schema_lines = [
            "Schema: column, dtype, non-null count, and unique values.",
            "Watch for: unexpected dtypes, many nulls, extremely low or high cardinality.",
            "",
        ]
        for c in df.columns:
            s = df[c]
            schema_lines.append(f"• {c:24s}  {str(s.dtype):12s}  non-null={s.notna().sum():,}  nulls={s.isna().sum():,}  unique={s.nunique(dropna=True):,}")
        add_text_page("Schema", schema_lines)

        # Numeric stats with description
        if num_cols:
            desc = df[num_cols].describe(percentiles=[0.5]).T.rename(columns={"50%": "p50"})
            lines = [
                "Numeric stats: count, mean, std, min, p50, max for numeric columns.",
                "Watch for: zero std (constant columns), huge ranges, NaNs in count.",
                "",
            ]
            for c, row in desc.iterrows():
                lines.append(f"• {c:24s}  n={int(row['count']):8d}  mean={row['mean']:.4g}  std={row['std']:.4g}  min={row['min']:.4g}  p50={row['p50']:.4g}  max={row['max']:.4g}")
            add_text_page("Numeric stats", lines)

        # Correlation heatmap page with description
        if len(num_cols) >= 2:
            corr = df[num_cols].corr(numeric_only=True)
            fig, ax = plt.subplots(figsize=(8.5, 7))
            if sns is not None:
                sns.heatmap(corr, ax=ax, cmap="coolwarm", center=0, cbar=True)
            else:
                im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
                fig.colorbar(im, ax=ax)
                ax.set_xticks(range(len(corr.columns)))
                ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
                ax.set_yticks(range(len(corr.columns)))
                ax.set_yticklabels(corr.columns, fontsize=7)
            ax.set_title("Correlation heatmap — strong |r| may indicate redundancy or leakage")
            fig.tight_layout()
            pdf_out.savefig(fig)
            plt.close(fig)

            # Top pairs table as text page
            iu = np.triu_indices_from(corr, k=1)
            pairs = (
                pd.DataFrame({
                    "col_A": corr.columns.values.repeat(len(corr)),
                    "col_B": np.tile(corr.columns, len(corr)),
                    "corr": corr.values.flatten(),
                })
                .loc[lambda x: x.index.isin([i * len(corr) + j for i, j in zip(*iu)])]
                .assign(corr=lambda x: x["corr"].round(3))
                .sort_values("corr", key=lambda s: s.abs(), ascending=False)
                .head(12)
            )
            lines = ["Top correlation pairs (Pearson). Watch for values near ±1.", ""]
            for _, r in pairs.iterrows():
                lines.append(f"• {r['col_A']} ↔ {r['col_B']}: corr={r['corr']:+.3f}")
            add_text_page("Top correlations", lines)

        # Expectations traffic-light summary (inline defaults)
        expectations = {
            "price": {"min": 99.2, "max": 100.3},
            "clean_price": {"min": 99.58, "max": 100.04},
            "y_bps": {"min": -60, "max": 60},
            "y_h_bps": {"min": -60, "max": 60},
            "y_pi_ref_bps": {"min": -12, "max": 36},
            "accrued_interest": {"min": 0.0, "max": 6.2},
            "dv01_per_100": {"min": 0.002, "max": 0.105},
            "vendor_liq_score": {"min": 0.5, "max": 60},
        }
        rows = ["column          observed_min   observed_max   status"]
        for k, band in expectations.items():
            if k in df.columns:
                s = df[k].astype(float)
                obs_min = float(s.min()) if len(s) > 0 else float("nan")
                obs_max = float(s.max()) if len(s) > 0 else float("nan")
                ok = (obs_min >= band["min"]) and (obs_max <= band["max"]) if np.isfinite(obs_min) and np.isfinite(obs_max) else False
                status = "OK" if ok else ("WARN" if np.isfinite(obs_min) and np.isfinite(obs_max) else "N/A")
                rows.append(f"• {k:16s} {obs_min:12.3f}   {obs_max:12.3f}   {status}")
        add_text_page("Top checks — expected vs observed", rows)

        # Quick checks summary: splits and invariants
        try:
            lines = [
                "Quick checks: BUY/SELL split, ATS%, cap% and key invariants",
                "",
            ]
            if "side" in df.columns:
                counts = df["side"].astype(str).value_counts()
                n_buy = int(counts.get("BUY", 0)); n_sell = int(counts.get("SELL", 0))
                lines.append(f"BUY={n_buy:,}  SELL={n_sell:,}  (total={len(df):,})")
                if "y_bps" in df.columns:
                    try:
                        m = df.groupby(df["side"].astype(str))["y_bps"].mean()
                        buy_m = float(m.get("BUY", float("nan"))); sell_m = float(m.get("SELL", float("nan")))
                        ok = (np.isfinite(buy_m) and np.isfinite(sell_m) and (sell_m > buy_m))
                        lines.append(f"mean(y_bps|SELL)={sell_m:.3f}  mean(y_bps|BUY)={buy_m:.3f}  → {'OK' if ok else 'WARN'}")
                    except Exception:
                        pass
            if "ats" in df.columns:
                p = float(df["ats"].astype(bool).mean()) if len(df) > 0 else float("nan")
                band_ok = np.isfinite(p) and (0.05 <= p <= 0.20)
                lines.append(f"ats share={p:.2%}  expected [5%,20%] → {'OK' if band_ok else 'WARN'}")
            if {"trace_cap_indicator","reported_size","cap_threshold"}.issubset(df.columns):
                capped = df["trace_cap_indicator"].astype(bool)
                n_capped = int(capped.sum())
                bad_eq = int((df.loc[capped, "reported_size"] != df.loc[capped, "cap_threshold"]).sum()) if n_capped > 0 else 0
                lines.append(f"capped rows={n_capped:,}; reported_size==cap_threshold mismatches={bad_eq}")
            add_text_page("Quick checks", lines)
        except Exception:
            pass

        # Daily medians strip (market sanity)
        if "trade_dt" in df.columns:
            try:
                dfd = df.copy()
                dfd["trade_dt"] = pd.to_datetime(dfd["trade_dt"]).dt.date
                cols = [c for c in ["price","clean_price","y_bps"] if c in dfd.columns]
                if cols:
                    med = dfd.groupby("trade_dt")[cols].median()
                    fig, ax1 = plt.subplots(figsize=(8.5, 3.5))
                    ax2 = ax1.twinx() if "y_bps" in cols else None
                    if "price" in med.columns:
                        ax1.plot(med.index, med["price"], label="price", color="#4C78A8")
                    if "clean_price" in med.columns:
                        ax1.plot(med.index, med["clean_price"], label="clean_price", color="#72B7B2")
                    if "y_bps" in med.columns and ax2 is not None:
                        ax2.plot(med.index, med["y_bps"], label="y_bps", color="#F58518")
                        ax2.set_ylabel("y_bps (right)")
                    ax1.set_title("Daily medians: price/clean_price (left), y_bps (right)")
                    ax1.set_xlabel("trade_dt")
                    ax1.legend(loc="upper left")
                    fig.autofmt_xdate(rotation=45)
                    fig.tight_layout()
                    pdf_out.savefig(fig)
                    plt.close(fig)
            except Exception:
                pass

        # Portfolio-only correlations
        if "is_portfolio" in df.columns:
            try:
                sub = df.loc[df["is_portfolio"].astype(bool)].copy()
                nump = sub.select_dtypes(include=[np.number])
                if nump.shape[1] >= 2 and len(sub) > 0:
                    corrp = nump.corr(numeric_only=True)
                    fig, ax = plt.subplots(figsize=(8.0, 6.5))
                    if sns is not None:
                        sns.heatmap(corrp, ax=ax, cmap="coolwarm", center=0, cbar=True)
                    else:
                        im = ax.imshow(corrp.values, cmap="coolwarm", vmin=-1, vmax=1)
                        fig.colorbar(im, ax=ax)
                        ax.set_xticks(range(len(corrp.columns)))
                        ax.set_xticklabels(corrp.columns, rotation=90, fontsize=7)
                        ax.set_yticks(range(len(corrp.columns)))
                        ax.set_yticklabels(corrp.columns, fontsize=7)
                    ax.set_title("Correlation heatmap — portfolio-only")
                    fig.tight_layout()
                    pdf_out.savefig(fig)
                    plt.close(fig)
            except Exception:
                pass

        # Per-sector and per-rating boxplots for y_bps
        if "y_bps" in df.columns and ("sector" in df.columns or "rating" in df.columns):
            try:
                if "sector" in df.columns:
                    fig, ax = plt.subplots(figsize=(8.5, 3.5))
                    data = [df.loc[df["sector"] == s, "y_bps"].astype(float).dropna().values for s in sorted(df["sector"].dropna().unique())]
                    labels = sorted(df["sector"].dropna().unique())
                    ax.boxplot(data, labels=labels, showfliers=False)
                    ax.set_title("y_bps by sector")
                    ax.set_ylabel("bps")
                    fig.tight_layout()
                    pdf_out.savefig(fig)
                    plt.close(fig)
                if "rating" in df.columns:
                    fig, ax = plt.subplots(figsize=(8.5, 3.5))
                    # preserve rating order if possible
                    order = [r for r in ["AAA","AA","A","BBB","BB","B"] if r in set(df["rating"].dropna().unique())]
                    groups = order if order else sorted(df["rating"].dropna().unique())
                    data = [df.loc[df["rating"] == r, "y_bps"].astype(float).dropna().values for r in groups]
                    ax.boxplot(data, labels=groups, showfliers=False)
                    ax.set_title("y_bps by rating")
                    ax.set_ylabel("bps")
                    fig.tight_layout()
                    pdf_out.savefig(fig)
                    plt.close(fig)
            except Exception:
                pass

        # Distribution pages with per-plot descriptions
        # Numeric
        for c in num_cols:
            s = df[c].dropna().astype(float)
            fig, ax = plt.subplots(figsize=(8.5, 4))
            ax.hist(s.values, bins=bins)
            ax.set_title(f"Histogram: {c}")
            ax.set_xlabel(c)
            ax.set_ylabel("count")
            fig.tight_layout()
            pdf_out.savefig(fig)
            plt.close(fig)
            # caption
            cap = [
                f"Histogram of {c}.",
                "Look for: skewness, long tails (outliers), spikes at sentinels, and empty bins indicating gaps.",
            ]
            add_text_page(f"About: {c}", cap)

        # Datetime
        for c in dt_cols:
            s = df[c].dropna().astype("int64")
            counts, edges = np.histogram(s, bins=bins)
            fig, ax = plt.subplots(figsize=(8.5, 4))
            ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge")
            ax.set_title(f"Histogram: {c}")
            ax.set_xlabel(c)
            ax.set_ylabel("count")
            fig.tight_layout()
            pdf_out.savefig(fig)
            plt.close(fig)
            add_text_page(f"About: {c}", [
                f"Time distribution for {c}.",
                "Look for: unexpected inactive periods, clustering at single timestamps, or timezone artifacts.",
            ])

        # Bool
        for c in bool_cols:
            s = df[c].dropna().astype(int)
            counts, _ = np.histogram(s, bins=[-0.5, 0.5, 1.5])
            fig, ax = plt.subplots(figsize=(8.5, 3))
            ax.bar([0, 1], counts, align="center")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["False", "True"])
            ax.set_title(f"Histogram: {c}")
            ax.set_ylabel("count")
            fig.tight_layout()
            pdf_out.savefig(fig)
            plt.close(fig)
            add_text_page(f"About: {c}", [
                f"Boolean distribution for {c}.",
                "Look for: extremely imbalanced classes which may harm model learning or indicate flags that are almost constant.",
            ])

    return pdf_path


@app.command()
def app_main(
    path: Path = typer.Argument(..., help="Path to a parquet or CSV file"),
    columns: Optional[str] = typer.Option(None, help="Comma-separated list of columns to include"),
    preview_rows: int = typer.Option(5, help="Show first N rows preview (0 to skip)"),
    topk: int = typer.Option(10, help="Top-K values for categorical distributions"),
    bins: int = typer.Option(20, help="Bins for numeric histograms"),
    correlations: bool = typer.Option(True, help="Show top correlated pairs for numeric columns"),
    corr_method: str = typer.Option("pearson", help="Correlation method: pearson|spearman|kendall"),
    top_pairs: int = typer.Option(10, help="How many correlated pairs to display"),
    plots: bool = typer.Option(False, help="Generate visualization PNGs and open a single consolidated window (off by default)"),
    outdir: Path = typer.Option(Path("reports/explore"), "--outdir", help="Directory to write images and reports"),
    pdf: bool = typer.Option(False, "--pdf", help="Write a single multi-page PDF report"),
    plot_outdir: Path = typer.Option(Path("reports/explore"), help="[DEPRECATED] Use --outdir instead"),
    show_components: bool = typer.Option(False, "--show-components", help="Reconcile y_bps against components and report residuals"),
    portfolio_splits: bool = typer.Option(False, "--portfolio-splits", help="Compare distributions for portfolio vs non-portfolio"),
    leakage_map: bool = typer.Option(False, "--leakage-map", help="List top correlations to y_bps and tag derived fields"),
    dv01_check: bool = typer.Option(False, "--dv01-check", help="Check DV01 arithmetic consistency"),
    html: Optional[Path] = typer.Option(None, "--html", help="Write a simple one-page HTML summary"),
):
    """
    Explore a Parquet/CSV file: schema, quick stats, distributions, correlations, and optional visuals.
    """
    console = Console()
    if not path.exists():
        raise typer.BadParameter(f"File not found: {path}")

    # Support CSV as well, reusing loader fallback if present
    cols = None
    if columns:
        cols = [c.strip() for c in columns.split(",") if c.strip()]
    # Try to read parquet first; if fails, try CSV
    try:
        df = _load_parquet(path, cols)
    except Exception:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path, usecols=cols if cols else None)
        else:
            # fallback to pandas auto CSV
            try:
                df = pd.read_csv(path, usecols=cols if cols else None)
            except Exception as e:
                raise e

    print(f"[bold]File:[/bold] {path} | rows={len(df):,} cols={len(df.columns)}")
    _print_schema(df, console)

    if preview_rows > 0:
        console.print("\n[bold]Preview[/bold]")
        console.print(df.head(preview_rows))

    console.print("\n[bold]Quick stats[/bold]")
    _print_quick_stats(df, console)

    console.print("\n[bold]Distributions[/bold]")
    _print_distributions(df, console, topk=topk, bins=bins)

    if correlations:
        console.print("\n[bold]Correlations[/bold]")
        _print_correlations(df, console, method=corr_method, top_pairs=top_pairs)

    # Resolve output directory, preferring --outdir over deprecated --plot_outdir
    outdir = outdir or plot_outdir
    outdir.mkdir(parents=True, exist_ok=True)

    if plots:
        console.print("\n[bold]Generating plots[/bold]")
        prefix = path.stem
        written = _save_plots(df, outdir, bins=bins, plot_prefix=prefix, show=True)
        if written:
            for p in written:
                console.print(f"  • wrote {p}")
        else:
            console.print("[yellow]No plots written[/yellow]")

    if pdf:
        console.print("\n[bold]Writing PDF report[/bold]")
        pdf_path = _write_pdf(df, outdir, path.stem, bins=bins)
        if pdf_path is not None:
            console.print(f"  • wrote {pdf_path}")
        else:
            console.print("[yellow]PDF not written (matplotlib not available)[/yellow]")

    # Optional diagnostics
    if show_components and {"y_bps","y_h_bps","y_eps_bps","y_pi_ref_bps","y_delta_port_bps"}.issubset(df.columns):
        console.print("\n[bold]Component reconciliation[/bold]")
        comp = df["y_pi_ref_bps"].astype(float) + df["y_h_bps"].astype(float) + df["y_eps_bps"].astype(float) + df["y_delta_port_bps"].fillna(0.0).astype(float)
        resid = (df["y_bps"].astype(float) - comp)
        console.print({
            "mean_abs": float(resid.abs().mean()),
            "std": float(resid.std(ddof=0)),
            "max_abs": float(resid.abs().max()),
            "p999_abs": float(resid.abs().quantile(0.999)),
        })

    if portfolio_splits and "is_portfolio" in df.columns:
        console.print("\n[bold]Portfolio vs Non-portfolio[/bold]")
        for col in [c for c in ["y_bps","reported_size","vendor_liq_score"] if c in df.columns]:
            grp = df.groupby(df["is_portfolio"].astype(bool))[col].describe()[["mean","std","50%","min","max"]]
            console.print(f"[italic]{col}[/italic]")
            console.print(grp)

        if "y_delta_port_bps" in df.columns:
            port = df.loc[df["is_portfolio"].astype(bool), "y_delta_port_bps"].astype(float)
            if len(port) > 0:
                console.print("y_delta_port_bps (portfolio only):", {
                    "mean": float(port.mean()),
                    "std": float(port.std(ddof=0)),
                    "p5": float(port.quantile(0.05)),
                    "p95": float(port.quantile(0.95)),
                })

    if leakage_map:
        console.print("\n[bold]Leakage map[/bold]")
        if "y_bps" in df.columns:
            num = df.select_dtypes(include=[np.number]).copy()
            corr = num.corr(method=corr_method)["y_bps"].sort_values(ascending=False)
            top = corr.drop(labels=["y_bps"], errors="ignore").head(top_pairs)
            derived = {"price","price_clean_exec","price_dirty_exec"}
            for k, v in top.items():
                tag = " ⚠ derived" if k in derived else ""
                console.print(f"{k:>24s}: {v: .3f}{tag}")

    if dv01_check and {"dv01_dollar","dv01_per_100","reported_size"}.issubset(df.columns):
        console.print("\n[bold]DV01 check[/bold]")
        calc = df["dv01_per_100"].astype(float) * (df["reported_size"].astype(float) / 100.0)
        y = df["dv01_dollar"].astype(float)
        resid = y - calc
        ss_res = float((resid ** 2).sum())
        ss_tot = float(((y - float(y.mean())) ** 2).sum())
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
        console.print({
            "max_abs_diff": float((resid.abs()).max()),
            "r2": float(r2),
        })

    if html is not None:
        try:
            html.parent.mkdir(parents=True, exist_ok=True)
            with open(html, "w", encoding="utf-8") as f:
                f.write("<html><head><meta charset='utf-8'><title>Explore report</title></head><body>")
                f.write(f"<h3>File: {path}</h3>")
                f.write(df.describe(include='all').to_html())
                if "y_bps" in df.columns:
                    f.write("<h4>Top correlations to y_bps</h4>")
                    num = df.select_dtypes(include=[np.number]).copy()
                    corr = num.corr(method=corr_method)["y_bps"].sort_values(ascending=False)
                    f.write(corr.to_frame(name="corr").head(25).to_html())
                f.write("</body></html>")
            console.print(f"\n[bold]HTML report:[/bold] wrote {html}")
        except Exception as e:
            console.print(f"[yellow]Failed to write HTML report: {e}[/yellow]")


# expose Typer app
app = app

if __name__ == "__main__":
    app()
