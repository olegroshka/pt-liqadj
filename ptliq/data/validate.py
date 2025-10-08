from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import numpy as np

@dataclass
class TableSpec:
    name: str
    required_cols: Dict[str, str]  # col -> kind (one of: "string","category","float","int","date","datetime","bool")
    unique_keys: List[str] | None = None

def _kind_of(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        # choose date vs datetime by resolution
        # (we won't enforce exactly at nanosecond level; datetime if any time component present)
        return "datetime"
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_integer_dtype(series):
        return "int"
    if pd.api.types.is_float_dtype(series):
        return "float"
    if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
        return "string"
    return "unknown"

def _limited(items: List[Any], n: int = 10) -> List[Any]:
    return items[:n]

def _check_table(path: Path, spec: TableSpec) -> Dict[str, Any]:
    report: Dict[str, Any] = {"table": spec.name, "path": str(path), "exists": path.exists()}
    if not path.exists():
        report["errors"] = [f"Missing table file: {path.name}"]
        report["passed"] = False
        return report

    df = pd.read_parquet(path)
    report["rows"] = int(len(df))
    report["columns"] = df.columns.tolist()
    errors: List[str] = []

    # required columns
    for col, kind in spec.required_cols.items():
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")
        else:
            observed = _kind_of(df[col])
            if kind in ("date", "datetime"):
                # allow either if user chose to export both as datetime
                if observed not in ("datetime",):
                    errors.append(f"Column {col}: expected {kind}, got {observed}")
            else:
                if observed != kind:
                    # allow 'string' for category etc. Keep simple now.
                    if not (kind == "string" and observed in ("string",)):
                        errors.append(f"Column {col}: expected {kind}, got {observed}")

    # non-null key columns
    if spec.unique_keys:
        for k in spec.unique_keys:
            if k in df.columns:
                nulls = int(df[k].isna().sum())
                if nulls > 0:
                    errors.append(f"Key column {k} has {nulls} nulls")

    # uniqueness
    if spec.unique_keys:
        if all(k in df.columns for k in spec.unique_keys):
            dup = int(df.duplicated(spec.unique_keys).sum())
            if dup > 0:
                errors.append(f"Duplicate key rows by {spec.unique_keys}: {dup}")

    report["errors"] = _limited(errors, 25)
    report["passed"] = len(errors) == 0
    return report

def validate_raw(rawdir: Path) -> Dict[str, Any]:
    """
    Validate the minimal raw schema: bonds.parquet + trades.parquet.
    Also check referential integrity (all trades.isin ∈ bonds.isin) and basic ranges.
    """
    rawdir = Path(rawdir)
    specs = [
        TableSpec(
            name="bonds",
            required_cols={
                "isin": "string",
                "issuer": "string",
                "sector": "string",
                "rating": "string",
                "issue_date": "datetime",   # stored as datetime for simplicity
                "maturity": "datetime",
                "coupon": "float",
                "amount_out": "float",
                "curve_bucket": "string",
            },
            unique_keys=["isin"],
        ),
        TableSpec(
            name="trades",
            required_cols={
                "ts": "datetime",
                "isin": "string",
                "side": "string",
                "size": "float",
                "price": "float",
                "is_portfolio": "bool",
            },
            unique_keys=None,
        ),
    ]

    table_reports: List[Dict[str, Any]] = []
    name_to_df: Dict[str, pd.DataFrame] = {}

    for spec in specs:
        path = rawdir / f"{spec.name}.parquet"
        trep = _check_table(path, spec)
        table_reports.append(trep)
        if trep.get("exists") and trep.get("passed"):
            name_to_df[spec.name] = pd.read_parquet(path)

    # Higher-level cross checks
    cross_errors: List[str] = []
    if "bonds" in name_to_df and "trades" in name_to_df:
        bonds = name_to_df["bonds"]
        trades = name_to_df["trades"]

        # referential integrity
        bad_isins = np.setdiff1d(trades["isin"].unique(), bonds["isin"].unique())
        if len(bad_isins) > 0:
            cross_errors.append(f"Trades reference unknown ISINs: {len(bad_isins)} (e.g. {', '.join(_limited(bad_isins.tolist(), 5))})")

        # basic value checks
        neg_sizes = int((trades["size"] <= 0).sum())
        if neg_sizes > 0:
            cross_errors.append(f"Non-positive trade sizes: {neg_sizes}")

        nonpos_prices = int((trades["price"] <= 0).sum())
        if nonpos_prices > 0:
            cross_errors.append(f"Non-positive trade prices: {nonpos_prices}")

        bad_sides = trades.loc[~trades["side"].isin(["BUY", "SELL"]), "side"].unique().tolist()
        if len(bad_sides) > 0:
            cross_errors.append(f"Unexpected side values: {bad_sides}")

        # maturity after issue
        if {"maturity", "issue_date"}.issubset(bonds.columns):
            wrong_order = int((pd.to_datetime(bonds["maturity"]) <= pd.to_datetime(bonds["issue_date"])).sum())
            if wrong_order > 0:
                cross_errors.append(f"maturity <= issue_date rows: {wrong_order}")

    passed_tables = all(tr.get("passed", False) for tr in table_reports)
    passed_cross = len(cross_errors) == 0
    return {
        "rawdir": str(rawdir),
        "passed": bool(passed_tables and passed_cross),
        "tables": table_reports,
        "cross_checks": _limited(cross_errors, 25),
    }
