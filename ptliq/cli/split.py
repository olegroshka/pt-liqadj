# ptliq/scripts/split.py  (or wherever your CLI lives today)
import json
from pathlib import Path
import typer

from ptliq.data.split import (
    compute_default_ranges,
    compute_auto_ranges,
    compute_rolling_ranges,
    write_ranges,
    count_rows_in_range,
    write_counts,
    write_masks,
)

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Time-aware dataset split generator")


def _run_split(
    rawdir: Path,
    trades_file: str,
    outdir: Path,
    filename: str,
    date_col: str,
    mode: str,
    train_end: str,
    val_end: str,
    val_days: int,
    test_days: int,
    embargo_days: int,
    n_folds: int,
    stride_days: int,
    embargo_train_val: int | None,
    embargo_val_test: int,
    write_counts_flag: bool,
    counts_filename: str,
    write_masks_flag: bool,
    masks_filename: str,
):
    trades_path = rawdir / trades_file

    if mode not in {"fixed", "auto", "rolling"}:
        raise typer.BadParameter("mode must be one of: fixed | auto | rolling")

    if mode == "fixed":
        split = compute_default_ranges(trades_path, train_end=train_end, val_end=val_end, date_col=date_col)
        path = write_ranges(split, outdir, filename=filename)
        counts = count_rows_in_range(trades_path, split, date_col=date_col)
        typer.echo(f"Wrote {path}  counts: {json.dumps(counts)}")
        if write_counts_flag:
            cpath = write_counts(trades_path, split, outdir, filename=counts_filename, date_col=date_col)
            typer.echo(f"Counts written to {cpath}")
        if write_masks_flag:
            mpath = write_masks(trades_path, split, outdir, filename=masks_filename, date_col=date_col)
            typer.echo(f"Masks written to {mpath}")
        return

    # derive effective embargo between train and val (back-compat with --embargo-days)
    etv = embargo_train_val if embargo_train_val is not None else embargo_days
    evt = embargo_val_test

    if mode == "auto" and n_folds == 0:
        try:
            split = compute_auto_ranges(
                trades_path,
                val_days=val_days,
                test_days=test_days,
                date_col=date_col,
                embargo_train_val=etv,
                embargo_val_test=evt,
            )
        except ValueError as e:
            # Graceful fallback to fixed split if auto sizing is impossible (e.g., too few days)
            typer.echo(f"Auto mode failed ({e}); falling back to fixed mode using --train-end/--val-end.")
            split = compute_default_ranges(trades_path, train_end=train_end, val_end=val_end, date_col=date_col)
        path = write_ranges(split, outdir, filename=filename)
        counts = count_rows_in_range(trades_path, split, date_col=date_col)
        typer.echo(f"Wrote {path}  counts: {json.dumps(counts)}")
        if write_counts_flag:
            cpath = write_counts(trades_path, split, outdir, filename=counts_filename, date_col=date_col)
            typer.echo(f"Counts written to {cpath}")
        if write_masks_flag:
            mpath = write_masks(trades_path, split, outdir, filename=masks_filename, date_col=date_col)
            typer.echo(f"Masks written to {mpath}")
        return

    # rolling (or auto + n_folds>0)
    folds = compute_rolling_ranges(
        trades_path,
        n_folds=max(n_folds, 3),
        val_days=val_days,
        test_days=test_days,
        embargo_days=etv,
        stride_days=(stride_days or None),
        date_col=date_col,
        embargo_val_test=evt,
    )
    path = write_ranges(folds, outdir, filename=filename)
    # Summarize counts
    summary = []
    for r in folds:
        summary.append({"fold_id": r.fold_id, **count_rows_in_range(trades_path, r, date_col=date_col)})
    typer.echo(f"Wrote {path}\nCounts per fold: {json.dumps(summary, indent=2)}")
    if write_counts_flag:
        cpath = write_counts(trades_path, folds, outdir, filename=counts_filename, date_col=date_col)
        typer.echo(f"Counts written to {cpath}")
    if write_masks_flag:
        mpath = write_masks(trades_path, folds, outdir, filename=masks_filename, date_col=date_col)
        typer.echo(f"Masks written to {mpath}")


# Subcommand: make
@app.command("make")
def app_main(
    rawdir: Path = typer.Option(Path("data/raw/sim"), help="Directory containing trades.parquet"),
    trades_file: str = typer.Option("trades.parquet", help="Trades file relative to rawdir"),
    outdir: Path = typer.Option(Path("data/interim/splits"), help="Output directory for split JSON"),
    filename: str = typer.Option("ranges.json", help="Output JSON filename"),
    date_col: str = typer.Option("ts", help="Time column to use (e.g. 'ts' or 'trade_dt')"),
    mode: str = typer.Option("auto", help="Split mode: fixed | auto | rolling"),
    # fixed-mode parameters
    train_end: str = typer.Option("2025-01-31", help="Train end date (YYYY-MM-DD) in fixed mode"),
    val_end: str = typer.Option("2025-03-31", help="Validation end date (YYYY-MM-DD) in fixed mode"),
    # auto / rolling parameters
    val_days: int = typer.Option(5, help="Validation window size in days (auto/rolling)"),
    test_days: int = typer.Option(5, help="Test window size in days (auto/rolling)"),
    embargo_days: int = typer.Option(1, help="Gap between train and val in days (legacy)", show_default=True),
    embargo_train_val: int | None = typer.Option(None, help="Embargo days between train and val (overrides --embargo-days)"),
    embargo_val_test: int = typer.Option(0, help="Embargo days between val and test"),
    # rolling-only
    n_folds: int = typer.Option(0, help="Number of rolling folds (0 = single split)"),
    stride_days: int = typer.Option(0, help="Stride between folds in days (0 = test_days)"),
    # outputs
    write_counts_flag: bool = typer.Option(False, "--write-counts/--no-write-counts", help="Also write counts.json"),
    counts_filename: str = typer.Option("counts.json", help="Counts JSON filename"),
    write_masks_flag: bool = typer.Option(False, "--write-masks/--no-write-masks", help="Also write masks.parquet"),
    masks_filename: str = typer.Option("masks.parquet", help="Masks Parquet filename"),
):
    _run_split(
        rawdir,
        trades_file,
        outdir,
        filename,
        date_col,
        mode,
        train_end,
        val_end,
        val_days,
        test_days,
        embargo_days,
        n_folds,
        stride_days,
        embargo_train_val,
        embargo_val_test,
        write_counts_flag,
        counts_filename,
        write_masks_flag,
        masks_filename,
    )


# Allow calling without subcommand for backward compatibility
@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    rawdir: Path = typer.Option(Path("data/raw/sim"), help="Directory containing trades.parquet"),
    trades_file: str = typer.Option("trades.parquet", help="Trades file relative to rawdir"),
    outdir: Path = typer.Option(Path("data/interim/splits"), help="Output directory for split JSON"),
    filename: str = typer.Option("ranges.json", help="Output JSON filename"),
    date_col: str = typer.Option("ts", help="Time column to use (e.g. 'ts' or 'trade_dt')"),
    mode: str = typer.Option("auto", help="Split mode: fixed | auto | rolling"),
    train_end: str = typer.Option("2025-01-31", help="Train end date (YYYY-MM-DD) in fixed mode"),
    val_end: str = typer.Option("2025-03-31", help="Validation end date (YYYY-MM-DD) in fixed mode"),
    val_days: int = typer.Option(5, help="Validation window size in days (auto/rolling)"),
    test_days: int = typer.Option(5, help="Test window size in days (auto/rolling)"),
    embargo_days: int = typer.Option(1, help="Gap between train and val in days (legacy)"),
    embargo_train_val: int | None = typer.Option(None, help="Embargo days between train and val (overrides --embargo-days)"),
    embargo_val_test: int = typer.Option(0, help="Embargo days between val and test"),
    n_folds: int = typer.Option(0, help="Number of rolling folds (0 = single split)"),
    stride_days: int = typer.Option(0, help="Stride between folds in days (0 = test_days)"),
    write_counts_flag: bool = typer.Option(False, "--write-counts/--no-write-counts", help="Also write counts.json"),
    counts_filename: str = typer.Option("counts.json", help="Counts JSON filename"),
    write_masks_flag: bool = typer.Option(False, "--write-masks/--no-write-masks", help="Also write masks.parquet"),
    masks_filename: str = typer.Option("masks.parquet", help="Masks Parquet filename"),
):
    if ctx.invoked_subcommand is None:
        _run_split(
            rawdir,
            trades_file,
            outdir,
            filename,
            date_col,
            mode,
            train_end,
            val_end,
            val_days,
            test_days,
            embargo_days,
            n_folds,
            stride_days,
            embargo_train_val,
            embargo_val_test,
            write_counts_flag,
            counts_filename,
            write_masks_flag,
            masks_filename,
        )


if __name__ == "__main__":
    app()
