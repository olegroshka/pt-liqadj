import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from ptliq.cli.featurize import featurize_pyg


def test_pyg_packs_portfolio_context(tmp_path: Path):
    graph_dir = tmp_path / "graph"
    out_dir = tmp_path / "pyg"
    graph_dir.mkdir(parents=True)
    out_dir.mkdir(parents=True)

    # Minimal nodes table
    nodes = pd.DataFrame({
        "node_id": [0, 1, 2],
        "isin": ["A", "B", "C"],
        "issuer": ["ISS1", "ISS2", "ISS3"],
        "issuer_name": ["ISS1", "ISS2", "ISS3"],
        "sector": ["SEC1", "SEC1", "SEC2"],
        "sector_id": [0, 0, 1],
        "rating": ["A", "A", "BBB"],
        "rating_num": [3, 3, 4],
        "curve_bucket": ["3-5y", "3-5y", "7-10y"],
        "tenor_years": [4.0, 4.5, 8.0],
        "currency": ["USD", "USD", "USD"],
    })
    nodes.to_parquet(graph_dir / "graph_nodes.parquet", index=False)

    # Minimal edges table
    edges = pd.DataFrame({
        "src_id": [0, 1],
        "dst_id": [1, 2],
        "relation": ["SECTOR", "SECTOR"],
        "weight": [1.0, 1.0],
        "relation_id": [0, 0],
    })
    edges.to_parquet(graph_dir / "graph_edges.parquet", index=False)

    # Minimal edge_index bundle
    np.savez_compressed(graph_dir / "edge_index.npz",
                        edge_index=np.array([[0,1],[1,2]], dtype=np.int64),
                        edge_weight=np.array([1.0,1.0], dtype=np.float32),
                        relation_id=np.array([0,0], dtype=np.int64))

    # Portfolio lines: a single group with two nodes
    pl = pd.DataFrame({
        "pf_gid": [0, 0],
        "portfolio_id": ["PF_20240715_X", "PF_20240715_X"],
        "trade_dt": pd.to_datetime(["2024-07-15", "2024-07-15"]).normalize(),
        "isin": ["A", "B"],
        "node_id": [0, 1],
        "w_dv01_abs_frac": [0.6, 0.4],
        "w_dv01_signed_frac": [0.6, 0.4],
        "rank_dv01_desc": [1.0, 2.0],
        "rank_pct": [0.0, 1.0],
        "dv01_abs": [60.0, 40.0],
    })
    pl.to_parquet(graph_dir / "portfolio_lines.parquet", index=False)

    # Run pyg assembly
    featurize_pyg.callback = None  # ensure Typer doesn't treat as CLI
    featurize_pyg(graph_dir=graph_dir, outdir=out_dir)

    # Check outputs
    ctx_path = out_dir / "portfolio_context.pt"
    idx_path = out_dir / "portfolio_index.parquet"
    meta_path = out_dir / "feature_meta.json"

    assert ctx_path.exists()
    assert idx_path.exists()
    assert meta_path.exists()

    ctx = torch.load(ctx_path)
    assert set(ctx.keys()) == {"port_nodes_flat","port_w_abs_flat","port_w_signed_flat","port_len","port_offsets"}
    # lengths/offsets consistent
    assert ctx["port_len"].dtype == torch.long
    assert ctx["port_offsets"].dtype == torch.long
    assert int(ctx["port_len"].sum()) == ctx["port_nodes_flat"].numel() == ctx["port_w_abs_flat"].numel() == ctx["port_w_signed_flat"].numel()

    # Index file aligns
    pidx = pd.read_parquet(idx_path).sort_values("pf_gid").reset_index(drop=True)
    assert len(pidx) == 1
    assert pidx.loc[0, "pf_gid"] == 0

    # Meta has portfolio_context section
    meta = json.loads(meta_path.read_text())
    assert "portfolio_context" in meta
    assert meta["portfolio_context"]["num_groups"] == 1
