from __future__ import annotations
from pathlib import Path
import json, zipfile, typer

app = typer.Typer(no_args_is_help=True)

@app.command()
def app(
    models_dir: Path = typer.Option(..., help="models/<run_id>/ with ckpt_gnn.pt"),
    run_id: str = typer.Option(...),
    graph_dir: Path = typer.Option(..., help="graph artifacts directory"),
    outdir: Path = typer.Option(Path("serving/packages")),
):
    src = Path(models_dir) / run_id
    outdir.mkdir(parents=True, exist_ok=True)
    zpath = outdir / f"{run_id}.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for name in ["ckpt_gnn.pt", "config.freeze.yaml", "metrics_val.json"]:
            p = src / name
            if p.exists(): z.write(p, arcname=name)
        # include graph artifacts
        for name in ["graph.json", "node_to_issuer.pt", "node_to_sector.pt", "issuer_groups.pt", "sector_groups.pt"]:
            p = Path(graph_dir) / name
            if p.exists(): z.write(p, arcname=f"graph/{name}")
    typer.echo(f"Packaged → {zpath}")

app = app