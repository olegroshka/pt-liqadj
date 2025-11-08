from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich import print

from ptliq.data.io import write_parquet
from ptliq.data.market import _infer_date_range, fetch_fred_series

app = typer.Typer(no_args_is_help=False, invoke_without_command=True)


@app.callback(invoke_without_command=True)
def _default(
    ctx: typer.Context,
    years: int = typer.Option(3, help="Number of past years to fetch if --start/--end not provided"),
    start: Optional[str] = typer.Option(None, help="Start date (YYYY-MM-DD). Overrides --years."),
    end: Optional[str] = typer.Option(None, help="End date (YYYY-MM-DD). Defaults to today."),
    out: Path = typer.Option(Path("data/raw/ig_oas.parquet"), help="Output parquet path"),
):
    if ctx.invoked_subcommand is None:
        fetch(years=years, start=start, end=end, out=out)


@app.command()
def fetch(
    years: int = typer.Option(5, help="Number of past years to fetch if --start/--end not provided"),
    start: Optional[str] = typer.Option(None, help="Start date (YYYY-MM-DD). Overrides --years."),
    end: Optional[str] = typer.Option(None, help="End date (YYYY-MM-DD). Defaults to today."),
    out: Path = typer.Option(Path("data/raw/ig_oas.parquet"), help="Output parquet path"),
):
    """
    Fetch ICE BofA US Corporate (IG) Option-Adjusted Spread (FRED: BAMLC0A0CM).
    """
    start_ts, end_ts = _infer_date_range(years, start, end)
    print(f"[bold]Fetching IG OAS (BAMLC0A0CM)[/bold] from [cyan]{start_ts.date()}[/cyan] to [cyan]{end_ts.date()}[/cyan]")
    try:
        df = fetch_fred_series("BAMLC0A0CM", start_ts, end_ts)
    except Exception as e:
        print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)
    path = write_parquet(df, out)
    print(f"[bold green]Done.[/bold green] rows={len(df):,} → {path}")


app = app

if __name__ == "__main__":
    app()
