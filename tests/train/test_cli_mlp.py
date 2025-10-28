from pathlib import Path
import pandas as pd
import numpy as np

from ptliq.data.simulate import SimParams, simulate
from ptliq.cli.gat_train import app_main as gat_train_main


def _make_graph_from_bonds(bonds: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    nodes = bonds.copy().reset_index(drop=True)
    nodes["node_id"] = np.arange(len(nodes), dtype=int)
    # minimal fields expected by _rebuild_data_from_tabular
    # sector_id, curve_bucket, currency present/derivable
    sector_levels = {s: i for i, s in enumerate(sorted(nodes["sector"].astype(str).unique()))}
    nodes["sector_id"] = nodes["sector"].astype(str).map(sector_levels).astype(int)
    # simple numeric placeholders
    nodes["rating_num"] = nodes["rating"].astype(str).map({r: i for i, r in enumerate(sorted(nodes["rating"].astype(str).unique()))}).astype(int)
    nodes["tenor_years"] = pd.to_numeric(nodes.get("tenor_years", 5.0), errors="coerce").fillna(5.0).astype(float)
    if "currency" not in nodes:
        nodes["currency"] = "USD"
    # keep required columns
    nodes = nodes[["isin","node_id","sector_id","curve_bucket","currency","rating_num","tenor_years"]]
    nodes.to_parquet(outdir / "graph_nodes.parquet", index=False)

    # minimal edges table (single self-edge to define relation_id=0)
    edges = pd.DataFrame({
        "src_id": [0],
        "dst_id": [0],
        "relation_id": [0],
        "weight": [1.0],
    })
    edges.to_parquet(outdir / "graph_edges.parquet", index=False)


def test_cli_supports_mlp_encoder(tmp_path: Path):
    # 1) simulate small dataset
    params = SimParams(n_bonds=120, n_days=6, providers=["P1"], seed=11, outdir=tmp_path)
    frames = simulate(params)
    bonds, trades = frames["bonds"], frames["trades"]
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "bonds.parquet").write_bytes(bonds.to_parquet(index=False))
    (raw_dir / "trades.parquet").write_bytes(trades.to_parquet(index=False))

    # 2) prepare minimal graph artifacts
    feat_dir = tmp_path / "features"
    _make_graph_from_bonds(bonds, feat_dir)

    # 3) run training via CLI entry with encoder_type=mlp
    outdir = tmp_path / "models_mlp"
    gat_train_main(
        features_run_dir=feat_dir,
        graph_dir=feat_dir,
        trades=raw_dir / "trades.parquet",
        outdir=outdir,
        config=None,
        ranges=None,
        max_epochs=1,
        batch_size=64,
        lr=1e-3,
        patience=1,
        seed=123,
        encoder_type="mlp",
        tb=False,
        tb_log_dir=None,
    )

    # 4) artifacts exist
    assert (outdir / "ckpt_liquidity.pt").exists(), "MLP training should produce a checkpoint"
    assert (outdir / "progress.jsonl").exists(), "progress log should be written"
