from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from typer.testing import CliRunner

from ptliq.cli.mvdgt_build import app as mvdgt_app
from ptliq.cli.featurize import featurize_pyg


runner = CliRunner()


def _write_min_graph(tmp: Path):
    gdir = tmp / "graph"; gdir.mkdir(parents=True)
    # nodes
    nodes = pd.DataFrame({
        'node_id': [0,1,2,3],
        'isin': ['A','B','C','D'],
        'issuer': ['I1','I1','I2','I3'],
        'issuer_name': ['I1','I1','I2','I3'],
        'sector': ['S1','S1','S2','S3'],
        'sector_id': [0,0,1,2],
        'rating': ['A','A','BBB','BB'],
        'rating_num': [3,3,4,5],
        'curve_bucket': ['3-5y','3-5y','7-10y','7-10y'],
        'tenor_years': [4.0,4.5,8.0,9.0],
        'currency': ['USD','USD','USD','USD']
    })
    nodes.to_parquet(gdir/"graph_nodes.parquet", index=False)
    # edges: SECTOR
    edges = pd.DataFrame({
        'src_id':[0,1,2],
        'dst_id':[1,2,3],
        'relation':['SECTOR','SECTOR','SECTOR'],
        'weight':[1.0,1.0,1.0],
        'relation_id':[0,0,0]
    })
    edges.to_parquet(gdir/"graph_edges.parquet", index=False)
    # npz stub
    np.savez_compressed(gdir/"edge_index.npz",
                        edge_index=np.array([[0,1],[1,2]], dtype=np.int64),
                        edge_weight=np.array([1.0,1.0], dtype=np.float32),
                        relation_id=np.array([0,0], dtype=np.int64))
    return gdir


def _write_min_trades(tmp: Path):
    tdir = tmp / "raw"; tdir.mkdir(parents=True)
    trades = pd.DataFrame({
        'isin': ['A','B','C','A','D'],
        'portfolio_id': ['P0','P0','P1','P2','P2'],
        'trade_dt': pd.to_datetime(['2024-01-02','2024-01-02','2024-01-03','2024-01-04','2024-01-04']).normalize(),
        'residual': [10.0, -5.0, 3.5, -2.0, 1.0]
    })
    p = tdir/"trades.parquet"; trades.to_parquet(p, index=False)
    return p


def test_root_cli_builds_dataset(tmp_path: Path):
    graph_dir = _write_min_graph(tmp_path)
    pyg_dir = tmp_path/"pyg"; pyg_dir.mkdir(parents=True)
    # Build PyG
    featurize_pyg.callback = None
    featurize_pyg(graph_dir=graph_dir, outdir=pyg_dir)

    outdir = tmp_path/"mvdgt/exp"
    trades_path = _write_min_trades(tmp_path)

    # Invoke root without subcommand, options at root
    result = runner.invoke(mvdgt_app, [
        "--trades-path", str(trades_path),
        "--graph-dir", str(graph_dir),
        "--pyg-dir", str(pyg_dir),
        "--outdir", str(outdir),
        "--split-train", "0.6",
        "--split-val", "0.2",
    ])
    assert result.exit_code == 0, result.output

    # Check artifacts
    samp_path = outdir/"samples.parquet"
    masks_path = outdir/"view_masks.pt"
    meta_path = outdir/"mvdgt_meta.json"
    assert samp_path.exists()
    assert masks_path.exists()
    assert meta_path.exists()

    # Validate masks length matches directed edges
    data = torch.load(pyg_dir/"pyg_graph.pt", map_location="cpu", weights_only=False)
    masks = torch.load(masks_path, map_location="cpu", weights_only=False)
    E = int(data.edge_index.size(1))
    for k, v in masks.items():
        assert int(v.numel()) == E

    # Validate split counts roughly match ratios
    df = pd.read_parquet(samp_path)
    n = len(df)
    n_tr = (df["split"]=="train").sum()
    n_va = (df["split"]=="val").sum()
    # Allow rounding differences at most 1
    assert abs(n_tr - int(0.6*n)) <= 1
    assert abs(n_va - int(0.2*n)) <= 1

    # meta contains files section
    meta = json.loads(meta_path.read_text())
    assert "files" in meta and "samples" in meta["files"]
