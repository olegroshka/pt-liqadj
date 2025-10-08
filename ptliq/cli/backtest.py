from __future__ import annotations
from pathlib import Path
from datetime import datetime
import typer
from rich import print

from ptliq.backtest.protocol import run_backtest

app = typer.Typer(no_args_is_help=True)

@app.command()
def app_main(
    features_dir: Path = typer.Option(Path("data/features")),
    run_id: str = typer.Option("exp001"),
    models_dir: Path = typer.Option(Path("models")),
    reports_dir: Path = typer.Option(Path("reports")),
):
    """
    Run backtest on test set: writes residuals.parquet + metrics.json under reports/<run_id>/backtest/<stamp>
    """
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    outdir = Path(reports_dir) / run_id / "backtest" / stamp
    outdir.mkdir(parents=True, exist_ok=True)
    metrics = run_backtest(features_dir, run_id, models_dir, outdir)
    print(f"[bold green]BACKTEST OK[/bold green] → {outdir}")
    print(f"  n={metrics['n']} test_mae_bps={metrics['overall']['mae_bps']:.3f}")

# expose Typer app
app = app

if __name__ == "__main__":
    app()
