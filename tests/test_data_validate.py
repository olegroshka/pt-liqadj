from pathlib import Path
from ptliq.data.simulate import SimParams, simulate
from ptliq.data.validate import validate_raw
import pandas as pd

def test_validate_on_sim(tmp_path: Path):
    # create a tiny simulated dataset
    frames = simulate(SimParams(n_bonds=30, n_days=2, providers=["P1"], seed=7, outdir=tmp_path))
    (tmp_path / "bonds.parquet").write_bytes(frames["bonds"].to_parquet(index=False))
    (tmp_path / "trades.parquet").write_bytes(frames["trades"].to_parquet(index=False))

    report = validate_raw(tmp_path)
    assert report["passed"], report

def test_validate_catches_bad_isin(tmp_path: Path):
    frames = simulate(SimParams(n_bonds=20, n_days=1, providers=["P1"], seed=8, outdir=tmp_path))
    bonds = frames["bonds"]
    trades = frames["trades"].copy()
    trades.loc[0, "isin"] = "FAKE123"  # break referential integrity
    bonds.to_parquet(tmp_path / "bonds.parquet", index=False)
    trades.to_parquet(tmp_path / "trades.parquet", index=False)

    report = validate_raw(tmp_path)
    assert not report["passed"]

    errs = [s.lower() for s in report.get("cross_checks", [])]
    assert errs and any("unknown isin" in e for e in errs)

