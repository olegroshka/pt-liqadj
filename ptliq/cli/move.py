from __future__ import annotations
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional

import typer
from rich import print

import pandas as pd

from ptliq.data.io import write_parquet

app = typer.Typer(no_args_is_help=True)


def _infer_date_range(years: int, start: Optional[str], end: Optional[str]) -> tuple[pd.Timestamp, pd.Timestamp]:
    tz_utc = timezone.utc
    if end:
        end_ts = pd.to_datetime(end).tz_localize(None)
    else:
        end_ts = pd.Timestamp(datetime.now(tz_utc)).tz_localize(None)
    if start:
        start_ts = pd.to_datetime(start).tz_localize(None)
    else:
        start_ts = end_ts - pd.Timedelta(days=365 * years + (years // 4))
    # floor/ceil to days
    return pd.to_datetime(start_ts.date()), pd.to_datetime((end_ts.date()))


def _download_move(start: pd.Timestamp, end: pd.Timestamp, interval: str = "1d") -> pd.DataFrame:
    try:
        import yfinance as yf  # type: ignore
    except Exception:
        raise RuntimeError(
            "yfinance is required to download MOVE index. Install it with: pip install yfinance"
        )

    df = yf.download(
        "^MOVE",
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    if df is None or df.empty:
        raise RuntimeError("No data received for ^MOVE from Yahoo Finance in the requested range.")

    # If yfinance returns a MultiIndex (Price, Ticker), collapse to the Price level
    if hasattr(df.columns, "nlevels") and getattr(df.columns, "nlevels", 1) > 1:
        try:
            df.columns = df.columns.get_level_values(0)
        except Exception:
            df.columns = [c[0] if isinstance(c, tuple) and len(c) > 0 else c for c in df.columns]

    # Capture original index (often datetime, sometimes unnamed)
    orig_index = df.index.copy()

    # Normalize schema: put date as a column, keep close and others if present
    df = df.reset_index()
    # Ensure consistent lowercase column names
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

    # Ensure we have a 'date' column
    if "date" not in df.columns:
        # Try any datetime-like column first
        dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        if dt_cols:
            df = df.rename(columns={dt_cols[0]: "date"})
        elif "index" in df.columns:
            # yfinance sometimes creates an 'index' column
            df["date"] = pd.to_datetime(df["index"], errors="coerce")
        else:
            # Fallback to the original index
            try:
                df["date"] = pd.to_datetime(orig_index, errors="coerce")
            except Exception:
                raise RuntimeError("Could not determine date column from Yahoo Finance data.")

    # If 'adj close' exists, prefer it as 'close'
    if "adj_close" in df.columns:
        df["close"] = df["adj_close"]

    # Keep a tidy selection from available columns only
    candidates = ["date", "open", "high", "low", "close", "volume"]
    keep = [c for c in candidates if c in df.columns]
    if "date" not in keep:
        # Last-resort: create from original index and try again
        df["date"] = pd.to_datetime(orig_index, errors="coerce")
        keep = [c for c in candidates if c in df.columns]

    df = df[keep].copy()
    # Make dates timezone-naive and sorted
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values("date").reset_index(drop=True)
    return df


@app.command()
def fetch(
    years: int = typer.Option(3, help="Number of past years to fetch if --start/--end not provided"),
    start: Optional[str] = typer.Option(None, help="Start date (YYYY-MM-DD). Overrides --years."),
    end: Optional[str] = typer.Option(None, help="End date (YYYY-MM-DD). Defaults to today."),
    out: Path = typer.Option(Path("data/raw/move.parquet"), help="Output parquet path"),
    interval: str = typer.Option("1d", help="Data interval for Yahoo Finance (e.g., 1d, 1wk, 1mo)"),
):
    """
    Fetch the ICE BofA MOVE Index (ticker ^MOVE on Yahoo Finance) and save it as a Parquet file.

    Examples:
      python -m ptliq.cli.move fetch --years 3 --out data/raw/move.parquet
      python -m ptliq.cli.move fetch --start 2022-01-01 --end 2025-10-18
    """
    start_ts, end_ts = _infer_date_range(years, start, end)
    print(f"[bold]Fetching ^MOVE[/bold] from [cyan]{start_ts.date()}[/cyan] to [cyan]{end_ts.date()}[/cyan] @ interval={interval}")

    try:
        df = _download_move(start_ts, end_ts, interval=interval)
    except Exception as e:
        print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)

    # Ensure schema matches OAS: [date, value]
    val_col = None
    for c in ("close", "adj_close"):
        if c in df.columns:
            val_col = c
            break
    if val_col is None:
        num_cols = [c for c in df.columns if c != "date"]
        if num_cols:
            val_col = num_cols[0]
        else:
            print("[red]No value column found in MOVE data.")
            raise typer.Exit(code=1)
    out_df = df[["date", val_col]].rename(columns={val_col: "value"})

    path = write_parquet(out_df, out)
    print(f"[bold green]Done.[/bold green] rows={len(out_df):,} → {path}")


# expose Typer app
app = app

if __name__ == "__main__":
    app()
