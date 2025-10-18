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
        # convert to int for binning (nanoseconds)
        s = s.view(np.int64)
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
    except Exception as e:
        print("[yellow]matplotlib not installed; skipping plots. Install with: pip install matplotlib seaborn[/yellow]")
        return written

    outdir.mkdir(parents=True, exist_ok=True)

    # Identify numeric columns once
    num_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
    ]

    # 1) Save individual files to disk (without leaving multiple figure windows open)
    heat_path = None
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
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
        fig_h.savefig(heat_path)
        plt.close(fig_h)
        written.append(heat_path)

    hist_series = []  # collect for composite figure
    for c in num_cols[:max_hists]:
        s = df[c].dropna()
        if s.empty:
            continue
        # Save individual histogram file
        fig_i = plt.figure(figsize=(5, 3))
        ax_i = fig_i.add_subplot(111)
        ax_i.hist(s, bins=bins, color="#4C78A8", edgecolor="white")
        ax_i.set_title(f"Histogram: {c}")
        ax_i.set_xlabel(c)
        ax_i.set_ylabel("count")
        hist_name = (f"{plot_prefix}__hist_{c}.png" if plot_prefix else f"hist_{c}.png")
        path = outdir / hist_name
        fig_i.tight_layout()
        fig_i.savefig(path)
        plt.close(fig_i)
        written.append(path)
        # store for composite display
        hist_series.append((c, s))

    # 2) Show a single consolidated window with all plots arranged in a scroll-friendly tall figure
    if show:
        try:
            n_h = len(hist_series)
            has_heat = heat_path is not None
            # layout: heatmap on top (optional) + grid of histograms below (3 per row)
            cols = 3 if n_h >= 3 else max(1, n_h)
            rows_hists = (n_h + cols - 1) // cols if n_h > 0 else 0
            rows_total = rows_hists + (1 if has_heat else 0)
            # Figure height scales with number of rows to allow scrolling in interactive backends
            fig_height = 3.0 * rows_total + (2.0 if has_heat else 0.0)
            fig = plt.figure(figsize=(max(9, cols * 4), max(6, fig_height)))
            gs = gridspec.GridSpec(rows_total, cols, height_ratios=[2.0] + [1.0] * rows_hists if has_heat else [1.0] * rows_hists)

            idx_row = 0
            if has_heat:
                ax = fig.add_subplot(gs[0, :])
                # Recompute corr to draw into this figure to avoid reading from file
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


@app.command()
def app_main(
    path: Path = typer.Argument(..., help="Path to a parquet file"),
    columns: Optional[str] = typer.Option(None, help="Comma-separated list of columns to include"),
    preview_rows: int = typer.Option(5, help="Show first N rows preview (0 to skip)"),
    topk: int = typer.Option(10, help="Top-K values for categorical distributions"),
    bins: int = typer.Option(20, help="Bins for numeric histograms"),
    correlations: bool = typer.Option(True, help="Show top correlated pairs for numeric columns"),
    corr_method: str = typer.Option("pearson", help="Correlation method: pearson|spearman|kendall"),
    top_pairs: int = typer.Option(10, help="How many correlated pairs to display"),
    plots: bool = typer.Option(False, help="Generate visualization PNGs and open a single consolidated window (off by default)"),
    plot_outdir: Path = typer.Option(Path("reports/explore"), help="Directory to write plots when --plots is used"),
):
    """
    Explore a parquet file: print schema, quick stats, distributions, and correlations. Optional plots can be saved to disk.
    """
    console = Console()
    if not path.exists():
        raise typer.BadParameter(f"File not found: {path}")

    cols = None
    if columns:
        cols = [c.strip() for c in columns.split(",") if c.strip()]
    df = _load_parquet(path, cols)

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

    if plots:
        console.print("\n[bold]Generating plots[/bold]")
        prefix = path.stem
        written = _save_plots(df, plot_outdir, bins=bins, plot_prefix=prefix, show=True)
        if written:
            for p in written:
                console.print(f"  • wrote {p}")
        else:
            console.print("[yellow]No plots written[/yellow]")


# expose Typer app
app = app

if __name__ == "__main__":
    app()
