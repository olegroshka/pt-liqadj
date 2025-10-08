from __future__ import annotations
from pathlib import Path
from datetime import datetime
import typer
from rich import print

from ptliq.features.build import build_features

app = typer.Typer(no_args_is_help=True)

@app.command()
def app_main(
    rawdir: Path = typer.Option(Path("data/raw/sim"), help="Raw parquet dir"),
    splits: Path = typer.Option(..., help="Path to ranges.json from ptliq-split"),
    outdir: Path = typer.Option(Path("data/features"), help="Output base dir"),
    run_id: str = typer.Option("exp001", help="Run identifier"),
):
    """
    Build minimal model-ready features and write train/val/test parquet files.
    """
    frames = build_features(rawdir, splits)

    out = Path(outdir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    for k, df in frames.items():
        path = out / f"{k}.parquet"
        df.to_parquet(path, index=False)
        print(f"  • wrote {k}: {path} (rows={len(df)})")

    print(f"[bold green]FEATURES READY[/bold green] → {out}")

# expose Typer app
app = app

if __name__ == "__main__":
    app()
