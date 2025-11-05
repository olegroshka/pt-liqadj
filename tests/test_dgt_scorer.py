from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from ptliq.service.scoring import DGTScorer
from ptliq.cli.mvdgt_build import app_main as build_mvdgt_data
from ptliq.cli.mvdgt_train import app_main as train_mvdgt
from ptliq.cli.featurize import featurize_pyg


def _write_min_graph(tmp: Path) -> Path:
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
    import numpy as _np
    _np.savez_compressed(gdir/"edge_index.npz",
                        edge_index=_np.array([[0,1],[1,2]], dtype=_np.int64),
                        edge_weight=_np.array([1.0,1.0], dtype=_np.float32),
                        relation_id=_np.array([0,0], dtype=_np.int64))
    return gdir


def _write_min_trades(tmp: Path) -> Path:
    tdir = tmp / "raw"; tdir.mkdir(parents=True)
    trades = pd.DataFrame({
        'isin': ['A','B','C','A','D'],
        'portfolio_id': ['P0','P0','P1','P2','P2'],
        'trade_dt': pd.to_datetime(['2024-01-02','2024-01-02','2024-01-03','2024-01-04','2024-01-04']).normalize(),
        'residual': [10.0, -5.0, 3.5, -2.0, 1.0]
    })
    p = tdir/"trades.parquet"; trades.to_parquet(p, index=False)
    return p


@pytest.mark.smoke
def test_dgt_scorer_smoke(tmp_path: Path):
    # Build minimal artifacts end-to-end in a temp folder (no external env required)
    graph_dir = _write_min_graph(tmp_path)
    pyg_dir = tmp_path/"pyg"; pyg_dir.mkdir(parents=True)

    featurize_pyg.callback = None  # silence Typer
    featurize_pyg(graph_dir=graph_dir, outdir=pyg_dir)
    # Ensure DGTScorer can find graph_nodes.parquet adjacent to pyg_graph.pt (fallback path)
    import shutil as _shutil
    _shutil.copy(graph_dir/"graph_nodes.parquet", pyg_dir/"graph_nodes.parquet")

    outdir = tmp_path/"mvdgt/exp"; outdir.mkdir(parents=True)
    trades_path = _write_min_trades(tmp_path)

    build_mvdgt_data.callback = None
    build_mvdgt_data(trades_path=trades_path, graph_dir=graph_dir, pyg_dir=pyg_dir, outdir=outdir,
                     split_train=0.6, split_val=0.2)

    # quick train 1 epoch CPU for speed
    train_mvdgt.callback = None
    train_mvdgt(workdir=outdir, pyg_dir=pyg_dir, epochs=1, batch_size=4, device_str="cpu")

    # ensure artifacts exist
    assert (outdir/"ckpt.pt").exists()
    assert (outdir/"mvdgt_meta.json").exists()

    # pick 3 ISINs from the graph
    nodes = pd.read_parquet(graph_dir/"graph_nodes.parquet")
    isins = nodes["isin"].astype(str).head(3).tolist()
    rows = [
        {"isin": isins[0], "side": "BUY",  "size": 5_000_000,  "asof_date": "2024-05-06"},
        {"isin": isins[1], "side": "SELL", "size": 10_000_000, "asof_date": "2024-05-06"},
        {"isin": isins[2], "side": "BUY",  "size": 1_000_000,  "asof_date": "2024-05-06"},
    ]

    scorer = DGTScorer.from_dir(outdir)
    y = scorer.score_many(rows)
    assert isinstance(y, np.ndarray) and y.shape == (3,)
    assert np.isfinite(y).all()
