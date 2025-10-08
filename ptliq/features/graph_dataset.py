from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch

from .graph_portfolio import GraphInputs, build_graph_inputs


@dataclass
class DateRange:
    start: pd.Timestamp
    end: pd.Timestamp  # inclusive


def _load_ranges(ranges_path: Path) -> Dict[str, DateRange]:
    js = pd.read_json(ranges_path)
    # expected keys: train, val, test with {"start": "...", "end": "..."}
    out: Dict[str, DateRange] = {}
    for split in ["train", "val", "test"]:
        d = js[split]
        out[split] = DateRange(
            start=pd.to_datetime(d["start"]).normalize(),
            end=pd.to_datetime(d["end"]).normalize(),
        )
    return out


def _load_raw(rawdir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    bonds = pd.read_parquet(Path(rawdir) / "bonds.parquet")
    trades = pd.read_parquet(Path(rawdir) / "trades.parquet")
    # normalize types
    bonds["issue_date"] = pd.to_datetime(bonds["issue_date"])
    bonds["maturity"] = pd.to_datetime(bonds["maturity"])
    trades["ts"] = pd.to_datetime(trades["ts"])
    return bonds, trades


def _slice_trades(trades: pd.DataFrame, dr: DateRange) -> pd.DataFrame:
    d = trades.copy()
    d_day = pd.to_datetime(d["ts"]).dt.normalize()
    mask = (d_day >= dr.start) & (d_day <= dr.end)
    return d.loc[mask].reset_index(drop=True)


def build_graph_inputs_for_split(
    rawdir: Path, ranges_path: Path, split: str, max_port_items: int = 128
) -> GraphInputs:
    """
    Build GraphInputs for a time range split (train/val/test).
    Portfolio = all trades on each day within the split; padding to max_port_items.
    """
    bonds, trades = _load_raw(Path(rawdir))
    ranges = _load_ranges(Path(ranges_path))
    if split not in ranges:
        raise KeyError(f"split {split!r} not in ranges file {ranges_path}")
    dtr = _slice_trades(trades, ranges[split])
    if len(dtr) == 0:
        # Return an empty but well-typed structure to avoid surprises in callers
        gi = build_graph_inputs(bonds.iloc[:0], trades.iloc[:0], max_port_items=max_port_items)
        return gi
    return build_graph_inputs(bonds, dtr, max_port_items=max_port_items)
