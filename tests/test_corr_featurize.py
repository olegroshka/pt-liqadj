import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


# --- Unit tests for correlation helpers ---

def test_daily_surprise_series_basic_properties():
    from ptliq.cli.featurize import _daily_surprise_series
    # Build synthetic trades for two bonds over 5 days with a price bump
    dates = pd.date_range("2025-01-01", periods=5, freq="D")
    rows = []
    for d in dates:
        rows.append({"trade_date": d, "isin": "A", "price": 100.0})
        rows.append({"trade_date": d, "isin": "A", "price": 101.0})
        rows.append({"trade_date": d, "isin": "B", "price": 200.0})
        rows.append({"trade_date": d, "isin": "B", "price": 199.0})
    trades = pd.DataFrame(rows)
    mat = _daily_surprise_series(trades)
    # Index should span the date range; columns include both isins
    assert "A" in mat.columns and "B" in mat.columns
    # No NaNs due to ffill and fillna(0)
    assert not mat.isna().any().any()
    # Columns are z-scored: mean approximately 0
    muA = float(mat["A"].mean()); muB = float(mat["B"].mean())
    assert abs(muA) < 1e-6 and abs(muB) < 1e-6


def test_corr_matrices_properties_and_separation():
    from ptliq.cli.featurize import _corr_matrix_pcc, _corr_matrix_mi

    rng = np.random.default_rng(0)
    T = 200
    # Column 0: base signal; Column 1: strongly dependent on 0; Column 2: independent noise
    a = rng.standard_normal(T)
    b = a + 0.05 * rng.standard_normal(T)
    c = rng.standard_normal(T)
    W = np.stack([a, b, c], axis=1).astype(np.float32)

    # PCC matrix
    C = _corr_matrix_pcc(W)
    assert C.shape == (3, 3)
    # Symmetry and diag zeros
    assert np.allclose(C, C.T, atol=1e-6)
    assert np.allclose(np.diag(C), 0.0)
    # In [0,1]
    assert (C >= -1e-6).all() and (C <= 1.0 + 1e-6).all()
    # Corr(a,b) should be high; corr(a,c) much lower
    assert C[0, 1] > 0.9
    assert C[0, 2] < 0.3

    # MI matrix (normalized to [0,1])
    MI = _corr_matrix_mi(W, n_bins=16)
    assert MI.shape == (3, 3)
    assert np.allclose(MI, MI.T, atol=1e-6)
    assert np.allclose(np.diag(MI), 0.0)
    assert (MI >= -1e-6).all() and (MI <= 1.0 + 1e-6).all()
    # Dependent pair (a,b) should have larger MI than (a,c)
    assert MI[0, 1] > MI[0, 2]


# --- Integration-ish test for pyg packing of correlation relations ---

@pytest.mark.parametrize("use_parquet", [True, False])
def test_pyg_corr_mask_and_relations(tmp_path: Path, use_parquet: bool):
    from ptliq.cli.featurize import featurize_pyg

    graph_dir = tmp_path / "graph"
    outdir = tmp_path / "pyg"
    graph_dir.mkdir(parents=True, exist_ok=True)

    # Minimal nodes table
    nodes = pd.DataFrame(
        {
            "node_id": [0, 1, 2, 3],
            "isin": ["A", "B", "C", "D"],
            "issuer": ["I1", "I1", "I2", "I3"],
            "issuer_name": ["I1", "I1", "I2", "I3"],
            "sector": ["SEC1", "SEC1", "SEC2", "SEC3"],
            "sector_id": [0, 0, 1, 2],
            "rating": ["A", "A", "BBB", "BB"],
            "rating_num": [3, 3, 4, 5],
            "curve_bucket": ["2-5Y", "2-5Y", "5-10Y", ">10Y"],
            "tenor_years": [3.0, 4.0, 7.0, 12.0],
            "currency": ["USD", "USD", "EUR", "USD"],
        }
    )

    # Edges include correlation and non-correlation relations
    edges = pd.DataFrame(
        {
            "src_id": [0, 0, 1, 2],
            "dst_id": [1, 2, 3, 3],
            "relation": ["PCC_GLOBAL", "MI_LOCAL", "COTRADE_CO", "SECTOR"],
            "weight": [0.95, 0.55, 1.2, 0.3],
        }
    )

    # Relation id mapping as featurize.graph does (sorted by name)
    rels = sorted(edges["relation"].unique().tolist())
    rel2id = {r: i for i, r in enumerate(rels)}
    edges["relation_id"] = edges["relation"].map(rel2id)

    if use_parquet:
        nodes.to_parquet(graph_dir / "graph_nodes.parquet", index=False)
        edges.to_parquet(graph_dir / "graph_edges.parquet", index=False)
    else:
        nodes.to_csv(graph_dir / "graph_nodes.csv", index=False)
        edges.to_csv(graph_dir / "graph_edges.csv", index=False)

    # Create a dummy edge_index.npz (content is not used by featurize_pyg but must exist)
    np.savez(graph_dir / "edge_index.npz", src=np.array([0]), dst=np.array([1]))

    # Run pyg packer
    featurize_pyg.callback(graph_dir=graph_dir, outdir=outdir) if hasattr(featurize_pyg, "callback") else featurize_pyg(
        graph_dir=graph_dir, outdir=outdir
    )

    # Load saved Data
    # Torch 2.4+ may default to weights_only=True in some environments; force full pickle load
    data = torch.load(outdir / "pyg_graph.pt", weights_only=False)
    assert hasattr(data, "edge_index") and hasattr(data, "edge_type")

    # Corr mask exists and matches edge count
    assert hasattr(data, "corr_edge_mask")
    mask = data.corr_edge_mask
    assert mask.numel() == data.edge_index.size(1)
    # Should have at least some True entries (for PCC/MI)
    assert mask.sum().item() > 0

    # Meta contains relations list including PCC/MI
    meta = json.loads((outdir / "feature_meta.json").read_text())
    rel_list = meta.get("relations", [])
    assert any(r.startswith("PCC_") for r in rel_list)
    assert any(r.startswith("MI_") for r in rel_list)

    # The MI relation weights should be in [0,1] as per normalization
    # Retrieve edges back and check range for MI rows
    if use_parquet:
        edges_back = pd.read_parquet(graph_dir / "graph_edges.parquet")
    else:
        edges_back = pd.read_csv(graph_dir / "graph_edges.csv")
    mi_rows = edges_back[edges_back["relation"].str.startswith("MI_")]
    if not mi_rows.empty:
        assert (mi_rows["weight"] >= 0).all() and (mi_rows["weight"] <= 1.0).all()
