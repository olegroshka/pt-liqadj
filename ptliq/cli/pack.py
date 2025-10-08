from __future__ import annotations
from pathlib import Path
import zipfile
import typer
from rich import print

app = typer.Typer(no_args_is_help=True)

@app.command()
def app_main(
    models_dir: Path = typer.Option(Path("models")),
    run_id: str = typer.Option("exp001"),
    outdir: Path = typer.Option(Path("serving/packages")),
):
    """
    Package a trained model into a single zip file for serving.
    """
    src = Path(models_dir) / run_id
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    dst = outdir / f"{run_id}.zip"

    required = ["ckpt.pt", "feature_names.json", "scaler.json", "train_config.json"]
    for f in required:
        if not (src / f).exists():
            raise typer.BadParameter(f"Missing {f} under {src}")

    with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in required:
            z.write(src / f, arcname=f)

    print(f"[bold green]PACK OK[/bold green] → {dst}")
    print("  contents:", ", ".join(required))

app = app

if __name__ == "__main__":
    app()
