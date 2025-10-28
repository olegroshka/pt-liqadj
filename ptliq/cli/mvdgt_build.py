from __future__ import annotations

from pathlib import Path
import typer

# Reuse the dataset builder from features module
from ptliq.features.build_mvdgt_dataset import build as build_dataset

app = typer.Typer(no_args_is_help=True)


@app.callback(invoke_without_command=True)
def app_main(
    trades_path: Path = typer.Option(Path("data/raw/sim/trades.parquet"), help="Trades parquet with residual label"),
    graph_dir: Path = typer.Option(Path("data/graph"), help="Graph artifacts dir (nodes/edges/rel2id.json)"),
    pyg_dir: Path = typer.Option(Path("data/pyg"), help="PyG artifacts dir (pyg_graph.pt, market_index.parquet)"),
    outdir: Path = typer.Option(Path("data/mvdgt/exp001"), help="Output working directory for MV-DGT data"),
    split_train: float = typer.Option(0.70, help="Train split fraction (chronological)"),
    split_val: float = typer.Option(0.15, help="Validation split fraction (chronological)"),
):
    """Build MV-DGT dataset artifacts (samples + view masks)."""
    build_dataset(
        trades_path=trades_path,
        graph_dir=graph_dir,
        pyg_dir=pyg_dir,
        outdir=outdir,
        split_train=split_train,
        split_val=split_val,
    )


if __name__ == "__main__":
    app()
