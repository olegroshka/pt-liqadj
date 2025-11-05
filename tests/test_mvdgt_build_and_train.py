import pandas as pd
import numpy as np
import torch
from pathlib import Path

from ptliq.cli.mvdgt_build import app_main as build_mvdgt_data
from ptliq.cli.mvdgt_train import app_main as train_mvdgt
from ptliq.cli.featurize import featurize_pyg


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
    # edges: only SECTOR (struct view)
    edges = pd.DataFrame({
        'src_id':[0,1,2],
        'dst_id':[1,2,3],
        'relation':['SECTOR','SECTOR','SECTOR'],
        'weight':[1.0,1.0,1.0],
        'relation_id':[0,0,0]
    })
    edges.to_parquet(gdir/"graph_edges.parquet", index=False)
    # edge_index npz (content not used by featurize_pyg beyond existence)
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


def test_build_and_train_end_to_end(tmp_path: Path):
    graph_dir = _write_min_graph(tmp_path)
    pyg_dir = tmp_path/"pyg"; pyg_dir.mkdir(parents=True)

    # Build PyG artifacts from graph
    featurize_pyg.callback = None  # silence Typer
    featurize_pyg(graph_dir=graph_dir, outdir=pyg_dir)

    # Build MV-DGT samples and view masks
    outdir = tmp_path/"mvdgt/exp"
    outdir.mkdir(parents=True)
    trades_path = _write_min_trades(tmp_path)

    build_mvdgt_data.callback = None
    build_mvdgt_data(trades_path=trades_path, graph_dir=graph_dir, pyg_dir=pyg_dir, outdir=outdir,
                     split_train=0.6, split_val=0.2)

    # Check outputs exist
    assert (outdir/"samples.parquet").exists()
    assert (outdir/"view_masks.pt").exists()
    assert (outdir/"mvdgt_meta.json").exists()

    # Quick smoke train for 2 epochs on CPU
    train_mvdgt.callback = None
    train_mvdgt(workdir=outdir, pyg_dir=pyg_dir, epochs=2, batch_size=4, device_str="cpu")
    # Check checkpoint: standard filename only
    assert (outdir/"ckpt.pt").exists()
