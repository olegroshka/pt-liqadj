from pathlib import Path
import pandas as pd

from ptliq.data.simulate import SimParams, simulate
from ptliq.data.split import compute_default_ranges
from ptliq.features.build import build_features
from ptliq.data.split import write_ranges

def test_split_inclusive_and_non_overlapping(tmp_path: Path):
    frames = simulate(SimParams(n_bonds=50, n_days=4, providers=["P1"], seed=3, outdir=tmp_path))
    frames["bonds"].to_parquet(tmp_path / "bonds.parquet", index=False)
    frames["trades"].to_parquet(tmp_path / "trades.parquet", index=False)

    ranges = compute_default_ranges(tmp_path / "trades.parquet", "2025-01-03", "2025-01-04")
    rpath = write_ranges(ranges, tmp_path / "splits")
    out = build_features(tmp_path, rpath)

    # inclusive ends
    assert out["train"]["trade_date"].max().date().isoformat() == ranges.train["end"]
    assert out["val"]["trade_date"].max().date().isoformat() == ranges.val["end"]

    # non-overlapping rows
    ids = {k: set(zip(df["trade_date"], df["isin"])) for k, df in out.items()}
    assert ids["train"].isdisjoint(ids["val"])
    assert ids["val"].isdisjoint(ids["test"])
    assert ids["train"].isdisjoint(ids["test"])
