from __future__ import annotations
from pathlib import Path
import typer
import uvicorn
from ptliq.service.scoring import Scorer
from ptliq.service.app import create_app

app = typer.Typer(no_args_is_help=True)

@app.command()
def app_main(
    package: Path = typer.Option(..., help="Path to models/<run_id> or serving/packages/<run_id>.zip"),
    host: str = typer.Option("0.0.0.0"),
    port: int = typer.Option(8000),
    device: str = typer.Option("cpu"),
):
    """
    Serve the model with FastAPI. Accepts either a model directory or a packaged zip.
    """
    package = Path(package)
    if package.is_dir():
        scorer = Scorer.from_dir(package, device=device)
    else:
        scorer = Scorer.from_zip(package, device=device)
    api = create_app(scorer)
    uvicorn.run(api, host=host, port=port, log_level="info")

app = app

if __name__ == "__main__":
    app()
