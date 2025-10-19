from __future__ import annotations
from pathlib import Path
import json
from datetime import date

import numpy as np
import pandas as pd
import torch

from ptliq.cli.featurize import featurize_graph, featurize_pyg, _load_series


def _run_graph_and_pyg(tmp_path: Path) -> tuple[Path, Path]:
    graph_dir = tmp_path / "graph"; graph_dir.mkdir(parents=True, exist_ok=True)
    pyg_dir = tmp_path / "pyg"; pyg_dir.mkdir(parents=True, exist_ok=True)

    bonds = Path("data/raw/sim/bonds.parquet")
    trades = Path("data/raw/sim/trades.parquet")
    # Build graph (also writes market_features.parquet if raw market files exist)
    featurize_graph.callback = None  # silence Typer context if any
    featurize_graph(
        bonds=bonds,
        trades=trades,
        outdir=graph_dir,
        expo_scale=5e4,
        target_min=200,
        target_max=300,
        cotrade_q=0.85,
        cotrade_topk=20,
        issuer_topk=20,
        sector_topk=8,
        rating_topk=4,
        curve_topk=4,
        currency_topk=0,
    )

    # Build pyg and pack market context
    featurize_pyg(graph_dir=graph_dir, outdir=pyg_dir)
    return graph_dir, pyg_dir


def test_market_features_generated_and_meta(tmp_path: Path):
    graph_dir, _ = _run_graph_and_pyg(tmp_path)

    # Check artifacts exist
    mkt_pq = graph_dir / "market_features.parquet"
    mkt_meta = graph_dir / "market_meta.json"
    assert mkt_pq.exists(), "market_features.parquet should be created by featurize graph"
    assert mkt_meta.exists(), "market_meta.json should be created by featurize graph"

    df = pd.read_parquet(mkt_pq)
    assert "asof_date" in df.columns
    # Must include core columns
    for col in [
        "MOVE_lvl", "IG_OAS_bps", "HY_OAS_bps", "VIX_lvl",
        "MOVE_lvl_chg_1d", "MOVE_lvl_chg_5d", "MOVE_lvl_z_20d",
        "VIX_lvl_chg_1d", "VIX_lvl_chg_5d", "VIX_lvl_z_20d",
        "HYIG_spread_bps", "IG_OAS_chg_5d", "HY_OAS_chg_5d", "HYIG_chg_5d",
    ]:
        assert col in df.columns, f"missing expected market feature: {col}"

    # No NaNs after the policy fill
    assert not df.isna().any().any(), "Market features should have no NaNs after fills/derivatives"

    meta = json.loads(mkt_meta.read_text())
    assert meta["asof_key"] == "asof_date"
    assert set(meta["feature_names"]).issubset(set(df.columns) - {"asof_date"})


def test_market_context_packed_and_dates_align(tmp_path: Path):
    graph_dir, pyg_dir = _run_graph_and_pyg(tmp_path)

    # Load saved context
    ctx_path = pyg_dir / "market_context.pt"
    idx_path = pyg_dir / "market_index.parquet"
    meta_path = pyg_dir / "feature_meta.json"

    assert ctx_path.exists(), "market_context.pt should be saved"
    assert idx_path.exists(), "market_index.parquet should be saved"
    assert meta_path.exists(), "feature_meta.json should include market_context section"

    ctx = torch.load(ctx_path)
    assert "mkt_dates" in ctx and "mkt_feat" in ctx

    mkt = pd.read_parquet(graph_dir / "market_features.parquet").copy()
    mkt = mkt.sort_values("asof_date").reset_index(drop=True)
    feat_cols = [c for c in mkt.columns if c != "asof_date"]

    # Shapes
    assert ctx["mkt_feat"].shape[0] == len(mkt)
    assert ctx["mkt_feat"].shape[1] == len(feat_cols)
    assert ctx["mkt_dates"].shape[0] == len(mkt)
    assert ctx["mkt_dates"].dtype == torch.long

    # Date mapping check: mkt_dates are int64 nanoseconds since epoch derived from asof_date
    expected_ns = pd.to_datetime(mkt["asof_date"]).astype("int64").to_numpy()
    got_ns = ctx["mkt_dates"].cpu().numpy()
    np.testing.assert_array_equal(got_ns, expected_ns)

    # Feature meta consistency
    meta = json.loads(meta_path.read_text())
    mc = meta.get("market_context", {})
    assert mc.get("num_days") == int(ctx["mkt_dates"].shape[0])
    assert mc.get("num_features") == int(ctx["mkt_feat"].shape[1])
    assert mc.get("index_file") == "market_index.parquet"
    assert mc.get("feature_names") == feat_cols


def test__load_series_parses_int64_ns_dates(tmp_path: Path):
    # Craft a small Parquet with int64 ns timestamps in a 'date' column and a 'value' column
    ns = [
        1666224000000000000,
        1666310400000000000,
        1666569600000000000,
        1666656000000000000,
        1666742400000000000,
        1666828800000000000,
        1666915200000000000,
    ]
    vals = [155.61, 156.95, 154.02, 147.78, 140.20, 142.78, 144.60]
    df = pd.DataFrame({"date": pd.Series(ns, dtype="int64"), "value": vals})
    p = tmp_path / "sample.parquet"
    df.to_parquet(p, index=False)

    out = _load_series(p)
    # Ensure asof_date is parsed to Python date objects at midnight UTC
    assert out["asof_date"].dtype == "O"  # date objects
    assert list(out["asof_date"].head(3)) == [date(2022, 10, 20), date(2022, 10, 21), date(2022, 10, 24)]
    # Values preserved
    assert np.allclose(out["value"].to_numpy()[:3], np.array(vals[:3], dtype=float))
