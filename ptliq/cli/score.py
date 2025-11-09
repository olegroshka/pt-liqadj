from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import typer

from ptliq.service.scoring import MLPScorer

# Typer application for scoring packaged models from a Parquet file.
# Exposes `app` so tests can import and invoke it via Typer's CliRunner.
app = typer.Typer(no_args_is_help=True)


@app.command()
def app_main(
    package: Path = typer.Option(..., help="Path to a packaged model .zip produced by ptliq-pack"),
    input_path: Path = typer.Option(..., help="Input Parquet with feature columns"),
    output_path: Path = typer.Option(..., help="Output Parquet with a 'preds_bps' column"),
    device: str = typer.Option("cpu", help="cpu|cuda"),
):
    """Score rows from a Parquet file using a packaged MLP model.

    This minimal CLI matches the test expectations:
      - loads a model from `--package` (zip)
      - reads rows from `--input-path`
      - writes predictions to `--output-path` in column `preds_bps`
    """
    df = pd.read_parquet(input_path)
    rows = df.to_dict("records")

    scorer = MLPScorer.from_zip(package, device=device)
    preds = scorer.score_many(rows)

    out = df.copy()
    out["preds_bps"] = np.asarray(preds, dtype=np.float32)
    out.to_parquet(output_path, index=False)


# alias for tests importing `app`
app = app

if __name__ == "__main__":
    app()
