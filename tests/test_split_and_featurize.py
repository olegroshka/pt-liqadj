from pathlib import Path
import json
from ptliq.data.simulate import SimParams, simulate
from ptliq.data.split import compute_default_ranges, write_ranges
from ptliq.features.build import build_features
import pandas as pd

def test_split_and_features(tmp_path: Path):
    # simulate tiny dataset
    frames = simulate(SimParams(n_bonds=40, n_days=4, providers=["P1"], seed=9, outdir=tmp_path))
    bonds = frames["bonds"]; trades = frames["trades"]
    bonds.to_parquet(tmp_path / "bonds.parquet", index=False)
    trades.to_parquet(tmp_path / "trades.parquet", index=False)

    # split: train first 2 days, val next 1 day, rest test
    # simulator starts at 2025-01-02; so pick 2025-01-03 as train_end, 2025-01-04 as val_end
    ranges = compute_default_ranges(tmp_path / "trades.parquet", "2025-01-03", "2025-01-04")
    rpath = write_ranges(ranges, tmp_path / "splits")

    # build features
    out = build_features(tmp_path, rpath)
    assert set(out.keys()) == {"train","val","test"}
    # non-empty (at least some rows in train/val/test for 4 days)
    assert sum(len(df) for df in out.values()) > 0
    # columns present
    for split, df in out.items():
        assert {"ts","isin","trade_date","y_bps","f_size_log","f_side_buy","f_coupon","f_amount_log","f_sector_code","f_rating_code","f_curve_code","f_days_to_mty"} <= set(df.columns)
        # finite targets
        assert pd.api.types.is_numeric_dtype(df["y_bps"])
