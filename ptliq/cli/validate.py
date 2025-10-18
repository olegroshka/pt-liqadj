# ptliq/cli/validate.py
from __future__ import annotations
from pathlib import Path
import json
import logging
import typer
from rich import print
from datetime import datetime

from ptliq.utils.logging import setup_logging
from ptliq.data.validate import validate_raw

app = typer.Typer(no_args_is_help=True)

@app.command()
def app_main(
    rawdir: Path = typer.Option(Path("data/raw/sim"), help="Directory containing raw parquet tables"),
    outdir: Path = typer.Option(Path("data/interim/validated"), help="Where to write validation report"),
    fail_on_error: bool = typer.Option(True, help="Exit with non-zero code if validation fails"),
    loglevel: str = typer.Option("INFO", help="Log level (DEBUG|INFO|WARNING|ERROR|CRITICAL)"),
    config: Path = typer.Option(Path("configs/base.yaml"), help="YAML base config with optional validation settings"),
):
    """
    Validate raw data (schema, keys, referential integrity, arithmetic identities).
    Writes a JSON report with errors and warnings.
    """
    setup_logging(loglevel)
    logging.info("Validating rawdir=%s", rawdir)
    # Load optional config for validator rules (expected_null, allow-lists)
    try:
        from ptliq.utils.config import load_config as _load_cfg  # lazy import
        cfg = _load_cfg(config)
    except Exception:
        cfg = None
    report = validate_raw(rawdir, config=cfg)

    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"validation_report_{stamp}.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Summaries
    table_errs = sum(len(t.get("errors", [])) for t in report.get("tables", []))
    table_warns = sum(len(t.get("warnings", [])) for t in report.get("tables", []))
    cross = report.get("cross_checks", {})
    cross_errs = len(cross.get("errors", []))
    cross_warns = len(cross.get("warnings", []))
    print(f"[dim]Summary:[/dim] errors={table_errs + cross_errs} warnings={table_warns + cross_warns}")

    if report["passed"]:
        print(f"[bold green]VALIDATION PASSED[/bold green]  report: {outpath}")
        # Print details if there are any warnings or (unexpected) errors
        if (table_errs + cross_errs + table_warns + cross_warns) > 0:
            for t in report.get("tables", []):
                errs = t.get("errors", [])
                warns = t.get("warnings", [])
                if errs or warns:
                    print(f"  • table={t['table']} errors={errs} warnings={warns}")
            cx = report.get("cross_checks", {})
            if cx.get("errors"): print(f"  • cross errors={cx['errors']}")
            if cx.get("warnings"): print(f"  • cross warnings={cx['warnings']}")
        raise typer.Exit(code=0)
    else:
        print(f"[bold red]VALIDATION FAILED[/bold red]  report: {outpath}")
        for t in report.get("tables", []):
            if not t.get("passed", False):
                print(f"  • table={t['table']} errors={t.get('errors')} warnings={t.get('warnings')}")
        if report.get("cross_checks"):
            cx = report["cross_checks"]
            if cx.get("errors"): print(f"  • cross errors={cx['errors']}")
            if cx.get("warnings"): print(f"  • cross warnings={cx['warnings']}")
        raise typer.Exit(code=1 if fail_on_error else 0)

# expose Typer app
app = app

if __name__ == "__main__":
    app()
