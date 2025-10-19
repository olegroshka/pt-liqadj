from __future__ import annotations
from pathlib import Path
from typing import Optional

import typer
from rich import print

from ptliq.data.io import write_parquet
from ptliq.data.market import _infer_date_range, fetch_yahoo_single

app = typer.Typer(no_args_is_help=False, invoke_without_command=True)


@app.callback(invoke_without_command=True)
def _default(
    ctx: typer.Context,
    years: int = typer.Option(3, help="Number of past years to fetch if --start/--end not provided"),
    start: Optional[str] = typer.Option(None, help="Start date (YYYY-MM-DD). Overrides --years."),
    end: Optional[str] = typer.Option(None, help="End date (YYYY-MM-DD). Defaults to today."),
    out: Path = typer.Option(Path("data/raw/vix.parquet"), help="Output parquet path"),
    interval: str = typer.Option("1d", help="Data interval for Yahoo Finance (e.g., 1d, 1wk, 1mo)"),
):
    if ctx.invoked_subcommand is None:
        fetch(years=years, start=start, end=end, out=out, interval=interval)


@app.command()
def fetch(
    years: int = typer.Option(3, help="Number of past years to fetch if --start/--end not provided"),
    start: Optional[str] = typer.Option(None, help="Start date (YYYY-MM-DD). Overrides --years."),
    end: Optional[str] = typer.Option(None, help="End date (YYYY-MM-DD). Defaults to today."),
    out: Path = typer.Option(Path("data/raw/vix.parquet"), help="Output parquet path"),
    interval: str = typer.Option("1d", help="Data interval for Yahoo Finance (e.g., 1d, 1wk, 1mo)"),
):
    """
    Fetch CBOE Volatility Index VIX (Yahoo: ^VIX) and save it as a Parquet file.
    """
    start_ts, end_ts = _infer_date_range(years, start, end)
    print(
        f"[bold]Fetching ^VIX[/bold] from [cyan]{start_ts.date()}[/cyan] to [cyan]{end_ts.date()}[/cyan] @ interval={interval}"
    )
    try:
        df = fetch_yahoo_single("^VIX", start_ts, end_ts, interval=interval)
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
        # fallback to any numeric column except date
        num_cols = [c for c in df.columns if c != "date"]
        if num_cols:
            val_col = num_cols[0]
        else:
            print("[red]No value column found in VIX data.")
            raise typer.Exit(code=1)
    out_df = df[["date", val_col]].rename(columns={val_col: "value"})

    path = write_parquet(out_df, out)
    print(f"[bold green]Done.[/bold green] rows={len(out_df):,} → {path}")


app = app

if __name__ == "__main__":
    app()
