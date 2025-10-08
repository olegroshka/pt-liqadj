from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import json
import pandas as pd

@dataclass
class SplitRanges:
    train: dict
    val: dict
    test: dict

def _ts_to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.tz_localize(None).dt.normalize()

def compute_default_ranges(trades_path: Path, train_end: str, val_end: str) -> SplitRanges:
    df = pd.read_parquet(trades_path, columns=["ts"])
    d = _ts_to_date(df["ts"])
    dmin = d.min()
    # normalize ends to dates
    te = pd.to_datetime(train_end).normalize()
    ve = pd.to_datetime(val_end).normalize()
    assert te < ve, "train_end must be < val_end"
    # test end open
    far = pd.Timestamp("2100-01-01")

    return SplitRanges(
        train={"start": str(dmin.date()), "end": str(te.date())},
        val={"start": str((te + pd.Timedelta(days=1)).date()), "end": str(ve.date())},
        test={"start": str((ve + pd.Timedelta(days=1)).date()), "end": str(far.date())},
    )

def write_ranges(ranges: SplitRanges, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "ranges.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"train": ranges.train, "val": ranges.val, "test": ranges.test}, f, indent=2)
    return path

def read_ranges(path: Path) -> SplitRanges:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return SplitRanges(train=obj["train"], val=obj["val"], test=obj["test"])
