# ptliq/data/split.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Union, Optional
from datetime import timedelta
import json
import pandas as pd


@dataclass
class SplitRanges:
    train: Dict[str, str]   # {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
    val: Dict[str, str]
    test: Dict[str, str]
    fold_id: Optional[int] = None  # populated for rolling splits


def _normalize_dates(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce")
    # robustly strip tz if present
    try:
        if s.dt.tz is not None:
            s = s.dt.tz_convert(None)
    except Exception:
        pass
    try:
        s = s.dt.tz_localize(None)
    except Exception:
        pass
    return s.dt.normalize()


def _read_date_series(trades_path: Union[str, Path], date_col: str = "ts") -> pd.Series:
    df = pd.read_parquet(trades_path, columns=[date_col])
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in {trades_path}")
    return _normalize_dates(df[date_col])


def compute_default_ranges(
    trades_path: Union[str, Path],
    train_end: str,
    val_end: str,
    date_col: str = "ts",
) -> SplitRanges:
    """
    Backwards-compatible fixed cutoffs: everything <= train_end -> train;
    (train_end, val_end] -> val; (val_end, +inf) -> test.
    Returns all date bounds as ISO strings for consistency.
    """
    s = _read_date_series(trades_path, date_col=date_col)
    start = s.min().date().isoformat()
    te = pd.Timestamp(train_end).normalize().date().isoformat()
    ve = pd.Timestamp(val_end).normalize().date().isoformat()
    val_start = (pd.Timestamp(train_end).normalize().date() + timedelta(days=1)).isoformat()
    test_start = (pd.Timestamp(val_end).normalize().date() + timedelta(days=1)).isoformat()
    far = pd.Timestamp("2100-01-01").date().isoformat()
    return SplitRanges(
        train={"start": start, "end": te},
        val={"start": val_start, "end": ve},
        test={"start": test_start, "end": far},
    )


def compute_auto_ranges(
    trades_path: Union[str, Path],
    val_days: int = 5,
    test_days: int = 5,
    embargo_days: Optional[int] = None,
    date_col: str = "ts",
    embargo_train_val: Optional[int] = None,
    embargo_val_test: int = 0,
) -> SplitRanges:
    """
    Derive split windows from the data that actually exist on disk with optional asymmetric embargoes.
    Defaults preserve prior behavior when only `embargo_days` was provided (interpreted as train→val embargo).

    Auto sizing follows:
      test = [Dmax - test_days + 1, Dmax]
      val  = [test.start - 1 - embargo_val_test - (val_days - 1), test.start - 1 - embargo_val_test]
      train= [Dmin, val.start - 1 - embargo_train_val]
    """
    s = _read_date_series(trades_path, date_col=date_col)
    dmin = s.min().normalize()
    dmax = s.max().normalize()

    etv = embargo_train_val if embargo_train_val is not None else (embargo_days if embargo_days is not None else 1)
    evt = max(0, int(embargo_val_test))

    # test window anchored at the end
    test_end = dmax
    test_start = test_end - pd.Timedelta(days=test_days - 1)

    # validation window before test, honoring val→test embargo
    val_end = test_start - pd.Timedelta(days=1 + evt)
    val_start = val_end - pd.Timedelta(days=val_days - 1)

    # training window ends etv days before val_start
    train_end = val_start - pd.Timedelta(days=1 + etv)

    if train_end < dmin:
        raise ValueError(
            f"Not enough history for auto split: min={dmin.date()} max={dmax.date()} "
            f"with val_days={val_days}, test_days={test_days}, embargo_train_val={etv}, embargo_val_test={evt}."
        )

    return SplitRanges(
        train={"start": dmin.date().isoformat(), "end": train_end.date().isoformat()},
        val={"start": val_start.date().isoformat(), "end": val_end.date().isoformat()},
        test={"start": test_start.date().isoformat(), "end": test_end.date().isoformat()},
    )


def compute_rolling_ranges(
    trades_path: Union[str, Path],
    n_folds: int = 3,
    val_days: int = 5,
    test_days: int = 5,
    embargo_days: int = 1,
    stride_days: Optional[int] = None,
    min_train_days: int = 14,
    date_col: str = "ts",
    embargo_val_test: int = 0,
) -> List[SplitRanges]:
    """
    Forward-chaining (expanding) rolling splits with optional embargoes.
    Fold k (counting back from the end):
      test = last `test_days` ending at (dmax - k*stride)
      val  = the `val_days` immediately preceding test, with an embargo between val→test of `embargo_val_test` days
      train = [dmin, val_start - 1 - embargo_days] (embargo between train→val)
    We stop early if not enough training history remains.
    """
    s = _read_date_series(trades_path, date_col=date_col)
    dmin = s.min().normalize()
    dmax = s.max().normalize()

    stride = stride_days or test_days  # by default, roll by the test window size
    evt = max(0, int(embargo_val_test))
    folds: List[SplitRanges] = []

    for k in range(n_folds):
        test_end = dmax - pd.Timedelta(days=k * stride)
        test_start = test_end - pd.Timedelta(days=test_days - 1)
        val_end = test_start - pd.Timedelta(days=1 + evt)
        val_start = val_end - pd.Timedelta(days=val_days - 1)
        train_end = val_start - pd.Timedelta(days=1 + embargo_days)
        train_start = dmin

        # Feasibility checks
        if train_end < train_start + pd.Timedelta(days=min_train_days - 1):
            break  # not enough history for this fold

        folds.append(
            SplitRanges(
                train={"start": train_start.date().isoformat(), "end": train_end.date().isoformat()},
                val={"start": val_start.date().isoformat(), "end": val_end.date().isoformat()},
                test={"start": test_start.date().isoformat(), "end": test_end.date().isoformat()},
                fold_id=k + 1,
            )
        )

    if not folds:
        raise ValueError("Could not construct any rolling folds with the requested parameters.")
    return folds


def write_ranges(
    ranges: Union[SplitRanges, List[SplitRanges]],
    outdir: Union[str, Path],
    filename: str = "ranges.json",
) -> Path:
    """
    Write a single split (`{"train":..., "val":..., "test":...}`) OR
    multiple folds (`{"folds": [ {train/val/test/fold_id}, ...] }`).
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / filename

    if isinstance(ranges, list):
        payload = {"folds": [asdict(r) for r in ranges]}
    else:
        payload = asdict(ranges)

    with path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def read_ranges(path: Union[str, Path]) -> Union[SplitRanges, List[SplitRanges]]:
    """
    Read either a single split or a list of folds from JSON.
    """
    path = Path(path)
    with path.open() as f:
        payload = json.load(f)

    def to_split(obj: Dict) -> SplitRanges:
        return SplitRanges(train=obj["train"], val=obj["val"], test=obj["test"], fold_id=obj.get("fold_id"))

    if "folds" in payload:
        return [to_split(obj) for obj in payload["folds"]]
    return to_split(payload)


def count_rows_in_range(
    trades_path: Union[str, Path],
    r: SplitRanges,
    date_col: str = "ts",
) -> Dict[str, int]:
    """
    Return counts per window for quick sanity checks.
    """
    df = pd.read_parquet(trades_path, columns=[date_col])
    dates = _normalize_dates(df[date_col]).dt.date

    def _mask(start: str, end: str):
        s = pd.to_datetime(start).date()
        e = pd.to_datetime(end).date()
        return (dates >= s) & (dates <= e)

    return {
        "train": int(_mask(r.train["start"], r.train["end"]).sum()),
        "val": int(_mask(r.val["start"], r.val["end"]).sum()),
        "test": int(_mask(r.test["start"], r.test["end"]).sum()),
    }


def write_counts(
    trades_path: Union[str, Path],
    ranges: Union[SplitRanges, List[SplitRanges]],
    outdir: Union[str, Path],
    filename: str = "counts.json",
    date_col: str = "ts",
) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / filename

    if isinstance(ranges, list):
        payload = [
            {"fold_id": r.fold_id, **count_rows_in_range(trades_path, r, date_col=date_col)} for r in ranges
        ]
    else:
        payload = count_rows_in_range(trades_path, ranges, date_col=date_col)

    with path.open("w") as f:
        json.dump(payload, f, indent=2)
    return path


def write_masks(
    trades_path: Union[str, Path],
    ranges: Union[SplitRanges, List[SplitRanges]],
    outdir: Union[str, Path],
    filename: str = "masks.parquet",
    date_col: str = "ts",
) -> Path:
    """
    Write row-wise split labels based on the provided ranges.
    - Single split: columns = [date_col, "split"]
    - Multiple folds: columns = [date_col, "fold_id", "split"] (long format)
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / filename

    df = pd.read_parquet(trades_path, columns=[date_col]).copy()
    df[date_col] = _normalize_dates(df[date_col]).dt.date

    def label_rows(r: SplitRanges) -> pd.Series:
        t_s, t_e = pd.to_datetime(r.train["start"]).date(), pd.to_datetime(r.train["end"]).date()
        v_s, v_e = pd.to_datetime(r.val["start"]).date(), pd.to_datetime(r.val["end"]).date()
        te_s, te_e = pd.to_datetime(r.test["start"]).date(), pd.to_datetime(r.test["end"]).date()
        dates = df[date_col]
        lab = pd.Series(["none"] * len(df), index=df.index)
        lab = lab.mask((dates >= t_s) & (dates <= t_e), "train")
        lab = lab.mask((dates >= v_s) & (dates <= v_e), "val")
        lab = lab.mask((dates >= te_s) & (dates <= te_e), "test")
        return lab

    if isinstance(ranges, list):
        parts = []
        for r in ranges:
            part = pd.DataFrame({date_col: df[date_col], "fold_id": r.fold_id, "split": label_rows(r)})
            parts.append(part)
        out = pd.concat(parts, axis=0, ignore_index=True)
    else:
        out = pd.DataFrame({date_col: df[date_col], "split": label_rows(ranges)})

    out.to_parquet(path, index=False)
    return path
