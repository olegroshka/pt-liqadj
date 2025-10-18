# ptliq/cli/gnn_build_graph.py

from __future__ import annotations
from pathlib import Path
import json, typer
import pandas as pd
import torch

cli = typer.Typer(no_args_is_help=True)

@cli.command(name="build-graph")
def build_graph(
    bonds: Path = typer.Option(..., help="bonds.parquet with columns: isin, issuer, sector, rating"),
    outdir: Path = typer.Option(Path("data/graph"), help="Output directory for graph artifacts"),
    min_issuer_size: int = typer.Option(1),
    min_sector_size: int = typer.Option(1),
):
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(bonds)
    # maps
    isin2idx = {isin:i for i, isin in enumerate(df["isin"].tolist())}
    issuers = sorted(df["issuer"].unique().tolist())
    sectors = sorted(df["sector"].unique().tolist())
    issuer2idx = {v:i for i,v in enumerate(issuers)}
    sector2idx = {v:i for i,v in enumerate(sectors)}

    node_to_issuer = torch.tensor([issuer2idx[x] for x in df["issuer"].tolist()], dtype=torch.long)
    node_to_sector = torch.tensor([sector2idx[x] for x in df["sector"].tolist()], dtype=torch.long)

    issuer_groups = {i: torch.tensor(df.index[df["issuer"]==v].tolist(), dtype=torch.long) for v,i in issuer2idx.items()}
    sector_groups = {i: torch.tensor(df.index[df["sector"]==v].tolist(), dtype=torch.long) for v,i in sector2idx.items()}

    torch.save(node_to_issuer, outdir / "node_to_issuer.pt")
    torch.save(node_to_sector, outdir / "node_to_sector.pt")
    torch.save(issuer_groups, outdir / "issuer_groups.pt")
    torch.save(sector_groups, outdir / "sector_groups.pt")

    graph_meta = {
        "n_nodes": len(isin2idx),
        "issuer_card": len(issuer2idx),
        "sector_card": len(sector2idx),
        "isin2idx": isin2idx,
        "issuer2idx": issuer2idx,
        "sector2idx": sector2idx,
    }
    (outdir / "graph.json").write_text(json.dumps(graph_meta))
    typer.echo(f"Graph built at {outdir}")

# expose a symbol named `app` that points to the Typer instance
app = cli
