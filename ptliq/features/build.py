from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict
import pandas as pd
import numpy as np
import json

def _load_raw(rawdir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    bonds = pd.read_parquet(rawdir / "bonds.parquet")
    trades = pd.read_parquet(rawdir / "trades.parquet")
    bonds["issue_date"] = pd.to_datetime(bonds["issue_date"])
    bonds["maturity"] = pd.to_datetime(bonds["maturity"])
    trades["ts"] = pd.to_datetime(trades["ts"])
    trades["trade_date"] = trades["ts"].dt.normalize()
    return bonds, trades

def _encode_cat(series: pd.Series) -> pd.Series:
    # stable codes from sorted categories
    cats = pd.Index(sorted(series.astype(str).unique()))
    mapper = {k: i for i, k in enumerate(cats)}
    return series.astype(str).map(mapper).astype(np.int32)

def _target_y_bps(trades: pd.DataFrame) -> pd.Series:
    # y = (price - median(price) per (trade_date, isin)) * 100
    med = trades.groupby(["trade_date", "isin"])["price"].transform("median")
    return (trades["price"] - med) * 100.0

def _days_to_maturity(trades_bonds: pd.DataFrame) -> pd.Series:
    return (pd.to_datetime(trades_bonds["maturity"]).dt.normalize() - trades_bonds["trade_date"]).dt.days.clip(lower=0)

def build_features(rawdir: Path, ranges_json: Path) -> Dict[str, pd.DataFrame]:
    bonds, trades = _load_raw(rawdir)

    # basic merge
    tb = trades.merge(bonds, on="isin", how="left", validate="many_to_one", suffixes=("", "_b"))

    # features
    tb["f_size_log"] = np.log1p(tb["size"].astype(float))
    tb["f_side_buy"] = (tb["side"] == "BUY").astype(np.int8)
    tb["f_coupon"] = tb["coupon"].astype(float)
    tb["f_amount_log"] = np.log1p(tb["amount_out"].astype(float))
    tb["f_sector_code"] = _encode_cat(tb["sector"])
    tb["f_rating_code"] = _encode_cat(tb["rating"])
    tb["f_curve_code"] = _encode_cat(tb["curve_bucket"]) 
    tb["f_days_to_mty"] = _days_to_maturity(tb).astype(np.int32)

    # additional flags
    # is_portfolio — true if sale_condition4 == 'P'
    tb["is_portfolio"] = tb.get("sale_condition4", pd.Series(index=tb.index)).eq("P").fillna(False).astype(bool)

    # target
    tb["y_bps"] = _target_y_bps(tb)

    # keep minimal columns (ts, isin, y, features)
    keep_cols = ["ts", "isin", "trade_date", "y_bps", "is_portfolio"] + [c for c in tb.columns if c.startswith("f_")]
    feat = tb[keep_cols].sort_values("ts").reset_index(drop=True)

    # apply split ranges
    with open(ranges_json, "r", encoding="utf-8") as f:
        ranges = json.load(f)

    def _in_range(df, rg):
        start = pd.to_datetime(rg["start"]).tz_localize(None)
        end = pd.to_datetime(rg["end"]).tz_localize(None)
        d = df["trade_date"]
        return df[(d >= start) & (d <= end)]

    return {
        "train": _in_range(feat, ranges["train"]),
        "val": _in_range(feat, ranges["val"]),
        "test": _in_range(feat, ranges["test"]),
    }
