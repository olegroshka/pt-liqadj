from __future__ import annotations
from pathlib import Path
import typer
from rich import print
from datetime import datetime

from ptliq.data.split import compute_default_ranges, write_ranges

app = typer.Typer(no_args_is_help=True)

@app.command()
def app_main(
    rawdir: Path = typer.Option(Path("data/raw/sim"), help="Folder with trades.parquet"),
    outdir: Path = typer.Option(Path("data/interim/splits"), help="Where to write split ranges"),
    train_end: str = typer.Option("2025-01-31", "--train-end", "--train_end", help="Inclusive train end (YYYY-MM-DD)"),
    val_end: str   = typer.Option("2025-03-31", "--val-end", "--val_end", help="Inclusive validation end (YYYY-MM-DD)"),
):
    """Create time-based split ranges (train/val/test) using trades.ts."""
    trades_path = rawdir / "trades.parquet"
    ranges = compute_default_ranges(trades_path, train_end=train_end, val_end=val_end)

    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    outpath = write_ranges(ranges, Path(outdir) / stamp)
    print(f"[bold green]SPLITS WRITTEN[/bold green] → {outpath}")

# expose Typer app
app = app

if __name__ == "__main__":
    app()
