from pathlib import Path
from ptliq.data.simulate import SimParams, simulate
from ptliq.data.split import compute_default_ranges, write_ranges
from ptliq.features.build import build_features

def test_no_temporal_leakage(tmp_path: Path):
    frames = simulate(SimParams(n_bonds=40, n_days=4, providers=["P1"], seed=5, outdir=tmp_path))
    frames["bonds"].to_parquet(tmp_path / "bonds.parquet", index=False)
    frames["trades"].to_parquet(tmp_path / "trades.parquet", index=False)

    r = compute_default_ranges(tmp_path / "trades.parquet", "2025-01-03", "2025-01-04")
    rpath = write_ranges(r, tmp_path / "splits")
    out = build_features(tmp_path, rpath)

    assert out["train"]["trade_date"].max() < out["val"]["trade_date"].min()
    assert out["val"]["trade_date"].max() < out["test"]["trade_date"].min()
