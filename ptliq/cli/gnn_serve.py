from __future__ import annotations
import typer
cli = typer.Typer(no_args_is_help=True)

@cli.command()
def app(
    package: str = typer.Option(...),
    host: str = typer.Option("0.0.0.0"),
    port: int = typer.Option(8080),
    device: str = typer.Option("cpu"),
):
    # Thin wrapper to your FastAPI app once we wire GNN scorer.
    raise SystemExit("ptliq-gnn-serve placeholder: we'll plug the GNN scorer after on-the-fly mapping lands.")

cli = app