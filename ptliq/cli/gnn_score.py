from __future__ import annotations
from pathlib import Path
import json, zipfile, tempfile, typer, torch, pandas as pd
from ptliq.cli.gnn_eval import _default_factory, _unwrap_state
from ptliq.model.utils import GraphInputs
from ptliq.model import resolve_device

app = typer.Typer(no_args_is_help=True)

@app.command()
def app(
    package: Path = typer.Option(..., help="zip from gnn-pack"),
    input_path: Path = typer.Option(..., help="parquet with the same columns used in gnn-build-dataset"),
    output_path: Path = typer.Option(..., help="parquet with a 'pred' column"),
    device: str = typer.Option("cpu"),
):
    with zipfile.ZipFile(package) as zf, tempfile.TemporaryDirectory() as td:
        zf.extractall(td)
        td = Path(td)
        # For simplicity here we require a pre-built test.pt next to input parquet in real usage.
        # You can extend this to rebuild GraphInputs on the fly by loading graph/ and mapping columns.
        raise SystemExit("For v0, use ptliq-gnn-eval on a dataset bundle; on-the-fly scoring is next.")

app = app