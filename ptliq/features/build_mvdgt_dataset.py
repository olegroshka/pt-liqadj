from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import typer

app = typer.Typer(no_args_is_help=True)

REL_STRUCT = {"ISSUER_SIBLING","SECTOR","RATING_NEAR","CURVE_BUCKET","CURRENCY"}
REL_PORT   = {"COTRADE_CO","COTRADE_X"}
REL_CG     = {"PCC_GLOBAL","MI_GLOBAL"}
REL_CL     = {"PCC_LOCAL","MI_LOCAL"}

def _load_edges_and_map(graph_dir: Path):
    edges_path = graph_dir / "graph_edges.parquet"
    if not edges_path.exists():
        edges_path = graph_dir / "graph_edges.csv"
    edges = pd.read_parquet(edges_path) if edges_path.suffix == ".parquet" else pd.read_csv(edges_path)
    relp = graph_dir / "rel2id.json"
    if relp.exists():
        rel2id = json.loads(relp.read_text())
        id2rel = {int(v): k for k, v in rel2id.items()}
    else:
        rel_names = sorted(pd.Series(edges["relation"], dtype=str).unique().tolist())
        rel2id = {k: i for i, k in enumerate(rel_names)}
        id2rel = {i: k for k, i in rel2id.items()}
    return edges, rel2id, id2rel

def _build_view_ids(id2rel: dict[int,str]) -> dict[str,set[int]]:
    views = {"struct": set(), "port": set(), "corr_global": set(), "corr_local": set()}
    for rid, name in id2rel.items():
        if name in REL_STRUCT:           views["struct"].add(rid)
        if name in REL_PORT:             views["port"].add(rid)
        if name in REL_CG:               views["corr_global"].add(rid)
        if name in REL_CL:               views["corr_local"].add(rid)
    return views

def _date_idx(trades: pd.DataFrame, mkt_index_path: Path) -> pd.Series:
    """Map trade dates to integer row_idx using market_index.parquet if present.
    If the index file is missing (market context was not built), create a minimal
    index from the trades themselves so downstream steps can proceed.
    """
    t = pd.to_datetime(trades["trade_dt"]).dt.normalize()
    if mkt_index_path.exists():
        idx = pd.read_parquet(mkt_index_path)  # cols: asof_date, row_idx
        idx["asof_date"] = pd.to_datetime(idx["asof_date"]).dt.normalize()
    else:
        # Build a minimal index from the unique trade dates
        uniq = (
            pd.DataFrame({"asof_date": pd.to_datetime(sorted(t.unique()))})
            .assign(row_idx=lambda df: np.arange(len(df), dtype=np.int64))
        )
        # Best-effort persist next to other PyG artifacts for reproducibility
        try:
            uniq.to_parquet(mkt_index_path, index=False)
        except Exception:
            pass
        idx = uniq
    return t.to_frame("asof_date").merge(idx, on="asof_date", how="left")["row_idx"].fillna(-1).astype(int)

def _attach_node_ids(trades: pd.DataFrame, nodes_path: Path) -> pd.Series:
    nodes = pd.read_parquet(nodes_path)[["isin","node_id"]]
    m = trades[["isin"]].merge(nodes, on="isin", how="left")["node_id"]
    return m.astype("Int64")

def _attach_pf_gid(trades: pd.DataFrame, port_lines_path: Path) -> pd.Series:
    if not port_lines_path.exists():
        return pd.Series([-1]*len(trades), dtype=int)
    pl = pd.read_parquet(port_lines_path)[["pf_gid","portfolio_id","trade_dt","isin"]].copy()
    pl["trade_dt"] = pd.to_datetime(pl["trade_dt"]).dt.normalize()
    t = trades[["portfolio_id","trade_dt","isin"]].copy()
    t["trade_dt"] = pd.to_datetime(t["trade_dt"]).dt.normalize()
    out = t.merge(pl, on=["portfolio_id","trade_dt","isin"], how="left")["pf_gid"].fillna(-1).astype(int)
    return out

def _make_view_masks(edge_type: torch.Tensor, id_sets: dict[str,set[int]]) -> dict[str, torch.Tensor]:
    out = {}
    for name, ids in id_sets.items():
        if len(ids) == 0:
            out[name] = torch.zeros_like(edge_type, dtype=torch.bool)
        else:
            out[name] = torch.isin(edge_type, torch.as_tensor(sorted(list(ids)), dtype=torch.long))
    # a convenience mask when you want “any correlation”
    out["corr_any"] = out["corr_global"] | out["corr_local"]
    return out

@app.command()
def build(
    trades_path: Path = typer.Option(Path("data/raw/sim/trades.parquet")),
    graph_dir:  Path = typer.Option(Path("data/graph")),
    pyg_dir:    Path = typer.Option(Path("data/pyg")),
    outdir:     Path = typer.Option(Path("data/mvdgt/exp001")),
    split_train: float = typer.Option(0.70),
    split_val:   float = typer.Option(0.15),
):
    outdir.mkdir(parents=True, exist_ok=True)

    # --- load graph + meta produced by your featurizer
    # Explicitly set weights_only=False for compatibility with full PyG Data objects under PyTorch security defaults
    data = torch.load(pyg_dir/"pyg_graph.pt", map_location="cpu", weights_only=False)   # contains edge_type (directed) etc.
    edges_df, rel2id, id2rel = _load_edges_and_map(graph_dir)
    id_sets = _build_view_ids(id2rel)
    view_masks = _make_view_masks(data.edge_type, id_sets)

    # persist view masks for the model
    torch.save({k: v.cpu() for k, v in view_masks.items()}, outdir/"view_masks.pt")

    # --- build per-trade samples
    trades = pd.read_parquet(trades_path)
    if "trade_dt" not in trades:  # fallbacks
        trades["trade_dt"] = pd.to_datetime(trades.get("exec_time", trades["ts"])).dt.normalize()

    samples = pd.DataFrame({
        "node_id": _attach_node_ids(trades, graph_dir/"graph_nodes.parquet"),
        "date_idx": _date_idx(trades, pyg_dir/"market_index.parquet"),
        "pf_gid": _attach_pf_gid(trades, graph_dir/"portfolio_lines.parquet"),
        "y": trades["residual"].astype(float),
        "side_sign": trades.get("side_sign", pd.Series([0.0]*len(trades))).astype(float),
        "log_size": trades.get("log_size", pd.Series([0.0]*len(trades))).astype(float),
    })
    samples = pd.concat([trades[["isin","portfolio_id","trade_dt"]].reset_index(drop=True), samples], axis=1)
    samples = samples.dropna(subset=["node_id","y"]).reset_index(drop=True)
    samples["node_id"] = samples["node_id"].astype(int)

    # chronological split
    samples = samples.sort_values("trade_dt").reset_index(drop=True)
    n = len(samples); n_tr = int(n*split_train); n_va = int(n*split_val)
    split = np.array(["test"]*n, dtype=object)
    split[:n_tr] = "train"
    split[n_tr:n_tr+n_va] = "val"
    samples["split"] = split

    samples.to_parquet(outdir/"samples.parquet", index=False)

    meta = {
        "views": {k: sorted([int(i) for i in v_ids]) for k, v_ids in id_sets.items()},
        "rel2id": rel2id,
        "files": {
            "pyg_graph": str((pyg_dir/"pyg_graph.pt").resolve()),
            "market_context": str((pyg_dir/"market_context.pt").resolve()),
            "portfolio_context": str((pyg_dir/"portfolio_context.pt").resolve()) if (pyg_dir/"portfolio_context.pt").exists() else None,
            "samples": str((outdir/"samples.parquet").resolve()),
            "view_masks": str((outdir/"view_masks.pt").resolve())
        }
    }
    (outdir/"mvdgt_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[OK] wrote {outdir/'samples.parquet'} and {outdir/'view_masks.pt'}")

if __name__ == "__main__":
    app()
