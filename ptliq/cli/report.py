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
    # Accept both: --make-html True/False and --make-html / --no-make-html
    make_html: str = typer.Option("True", help="Also write an HTML report (True/False) or use --make-html/--no-make-html"),
):
    """
    Generate plots for the latest backtest stamp under reports/<run_id>/backtest/<stamp>/figures
    and (optionally) write report.html next to metrics.json.
    """
    # --- tolerate both styles for boolean ---
    def _to_bool(v: str) -> bool:
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        if s in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "f", "no", "n", "off"}:
            return False
        # if user just passed --make-html (flag style), Typer may give "True" anyway; default to True
        return True

    make_html_bool = _to_bool(make_html)

    base = Path(reports_dir) / run_id / "backtest"
    if not base.exists():
        # Be robust: create a minimal report instead of failing
        run_dir = Path(reports_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        if make_html_bool:
            html = run_dir / "report.html"
            html.write_text(
                f"<!doctype html><html><body><h1>{run_id} Report</h1><p>No backtest found at {base}</p></body></html>",
                encoding="utf-8",
            )
            print(f"[bold yellow]No backtest found[/bold yellow]; wrote minimal HTML → {html}")
        else:
            print(f"[bold yellow]No backtest found[/bold yellow] under {base}")
        return

    # pick latest backtest stamp (if multiple)
    stamps = sorted([p for p in base.glob("*") if p.is_dir()])
    if not stamps:
        # Same graceful fallback
        run_dir = Path(reports_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        if make_html_bool:
            html = run_dir / "report.html"
            html.write_text(
                f"<!doctype html><html><body><h1>{run_id} Report</h1><p>No backtest stamps found in {base}</p></body></html>",
                encoding="utf-8",
            )
            print(f"[bold yellow]No backtest stamps[/bold yellow]; wrote minimal HTML → {html}")
        else:
            print(f"[bold yellow]No backtest stamps[/bold yellow] under {base}")
        return

    latest = stamps[-1]

    figs = render_report(latest)
    print(f"[bold green]REPORT READY[/bold green] → {latest/'figures'}")
    for k, v in figs.items():
        print(f"  • {k}: {v}")

    if make_html_bool:
        html = render_html(latest, title=f"{run_id} backtest")
        print(f"[bold]HTML[/bold] → {html}")


app = app

if __name__ == "__main__":
    app()
