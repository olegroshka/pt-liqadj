from __future__ import annotations
from pathlib import Path
import typer
from rich import print

from ptliq.viz.report import render_report, render_html

app = typer.Typer(no_args_is_help=True)

@app.command()
def app_main(
    reports_dir: Path = typer.Option(Path("reports"), help="Base reports directory"),
    run_id: str = typer.Option("exp001", help="Run id"),
    make_html: bool = typer.Option(True, help="Also write an HTML report"),  # use --make-html / --no-make-html
):
    """
    Generate plots for the latest backtest stamp under reports/<run_id>/backtest/<stamp>/figures
    and (optionally) write report.html next to metrics.json.
    """
    base = Path(reports_dir) / run_id / "backtest"
    stamps = sorted([p for p in base.glob("*") if p.is_dir()])
    if not stamps:
        raise typer.BadParameter(f"No backtest runs found under {base}")
    latest = stamps[-1]

    figs = render_report(latest)
    print(f"[bold green]REPORT READY[/bold green] → {latest/'figures'}")
    for k, v in figs.items():
        print(f"  • {k}: {v}")

    if make_html:
        html = render_html(latest, title=f"{run_id} backtest")
        print(f"[bold]HTML[/bold] → {html}")

app = app

if __name__ == "__main__":
    app()
