from __future__ import annotations
from pathlib import Path
import typer
from ptliq.viz.report import app as _report  # reuse your existing report CLI if present

app = typer.Typer(no_args_is_help=True)

@app.command()
def app(
    reports_dir: Path = typer.Option(Path("reports")),
    run_id: str = typer.Option("exp_gnn"),
    make_html: str = typer.Option("True"),
):
    # delegate to the tolerant report app you just patched
    _report.callback(reports_dir=reports_dir, run_id=run_id, make_html=make_html) if hasattr(_report, "callback") else _report(reports_dir, run_id, make_html)
