from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import json
import numpy as np
import pandas as pd

# Matplotlib is optional; skip plots if not installed
try:
    import matplotlib.pyplot as plt  # type: ignore
    _HAVE_MPL = True
except Exception:
    plt = None  # type: ignore
    _HAVE_MPL = False


def _summ(series: pd.Series) -> Dict[str, float | int]:
    if series is None or len(series) == 0:
        return {}
    q = series.quantile([0, 0.25, 0.5, 0.75, 1.0])
    return dict(
        min=int(q.iloc[0]),
        p25=float(q.iloc[1]),
        median=float(q.iloc[2]),
        mean=float(series.mean()),
        p75=float(q.iloc[3]),
        max=int(q.iloc[4]),
    )


def validate_graph_artifacts(nodes_path: Path, edges_path: Path, npz_path: Path, outdir: Path) -> Dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load nodes/edges from Parquet or CSV depending on suffix
    if nodes_path.suffix.lower() == ".parquet":
        nodes = pd.read_parquet(nodes_path)
    else:
        nodes = pd.read_csv(nodes_path)

    if edges_path.suffix.lower() == ".parquet":
        edges = pd.read_parquet(edges_path)
    else:
        edges = pd.read_csv(edges_path)

    _ = np.load(npz_path)

    problems = []
    if (edges["src_id"] > edges["dst_id"]).any():
        problems.append("edges not canonical (src_id > dst_id present)")
    if (edges["src_id"] == edges["dst_id"]).any():
        problems.append("self-loops present")
    if edges[["src_id", "dst_id", "relation"]].duplicated().any():
        problems.append("duplicate relation rows present")

    rel_deg_counts = pd.concat([edges["src_id"], edges["dst_id"]]).value_counts().sort_index()
    pairs = edges[["src_id", "dst_id"]].drop_duplicates()
    uniq_deg_counts = pd.concat([pairs["src_id"], pairs["dst_id"]]).value_counts().sort_index()

    report = {
        "n_nodes": int(len(nodes)),
        "n_edge_rows": int(len(edges)),
        "relations": edges["relation"].value_counts().sort_index().to_dict(),
        "relation_degree_stats": _summ(rel_deg_counts) if len(rel_deg_counts) else {},
        "unique_degree_stats": _summ(uniq_deg_counts) if len(uniq_deg_counts) else {},
        "problems": problems,
    }
    (outdir / "graph_report.json").write_text(json.dumps(report, indent=2))

    if _HAVE_MPL:
        plt.figure()
        plt.hist(rel_deg_counts.values, bins=20)
        plt.title("Relation-degree histogram")
        plt.xlabel("relation-degree")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(outdir / "relation_degree_hist.png", dpi=160)
        plt.close()

        plt.figure()
        plt.hist(uniq_deg_counts.values, bins=20)
        plt.title("Unique-neighbor degree histogram")
        plt.xlabel("unique-degree")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(outdir / "unique_degree_hist.png", dpi=160)
        plt.close()

    # top hubs
    top = (
        uniq_deg_counts.sort_values(ascending=False)
        .head(20)
        .rename("unique_degree")
        .reset_index()
        .rename(columns={"index": "node_id"})
    )
    keep_cols = [c for c in ["node_id", "isin", "issuer_name", "sector", "rating", "curve_bucket"] if c in nodes.columns]
    top = top.merge(nodes[[c for c in keep_cols if c in nodes.columns]], on="node_id", how="left")
    top.to_csv(outdir / "top_hubs.csv", index=False)

    return report
