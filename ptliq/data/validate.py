# ptliq/data/validate.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ----------------------------
# Schema & rule infrastructure
# ----------------------------
@dataclass
class TableSpec:
    name: str
    required_cols: Dict[str, str]  # col -> kind ("string","category","float","int","date","datetime","bool")
    unique_keys: List[str] | None = None

# accepted domain values (soft-checked; only warn if unexpected)
RATINGS = {"AAA","AA","A","BBB","BB","B"}
SECTORS = {"FIN","IND","UTIL","TEL","TECH"}
DAY_COUNTS = {"30/360","ACT/ACT"}

def _kind_of(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_bool_dtype(series): return "bool"
    if pd.api.types.is_integer_dtype(series): return "int"
    if pd.api.types.is_float_dtype(series): return "float"
    if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series): return "string"
    return "unknown"

def _limited(items: List[Any], n: int = 25) -> List[Any]:
    return items[:n]


# ---------------
# Table validators
# ---------------
def _check_table(path: Path, spec: TableSpec) -> Dict[str, Any]:
    report: Dict[str, Any] = {"table": spec.name, "path": str(path), "exists": path.exists()}
    if not path.exists():
        report["errors"] = [f"Missing table file: {path.name}"]
        report["warnings"] = []
        report["passed"] = False
        return report

    df = pd.read_parquet(path)
    report["rows"] = int(len(df))
    report["columns"] = df.columns.tolist()
    errors: List[str] = []
    warnings: List[str] = []

    # required columns + kind
    for col, kind in spec.required_cols.items():
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")
        else:
            observed = _kind_of(df[col])
            if kind in ("date", "datetime"):
                if observed not in ("datetime",):
                    errors.append(f"Column {col}: expected {kind}, got {observed}")
            else:
                if observed != kind:
                    # allow strings stored as object
                    if not (kind == "string" and observed in ("string",)):
                        errors.append(f"Column {col}: expected {kind}, got {observed}")

    # non-null key columns / uniqueness
    if spec.unique_keys:
        for k in spec.unique_keys:
            if k in df.columns:
                nulls = int(df[k].isna().sum())
                if nulls > 0:
                    errors.append(f"Key column {k} has {nulls} nulls")
        if all(k in df.columns for k in spec.unique_keys):
            dup = int(df.duplicated(spec.unique_keys).sum())
            if dup > 0:
                errors.append(f"Duplicate key rows by {spec.unique_keys}: {dup}")

    # lightweight domain hints (warn only; allow real-world values)
    if spec.name == "bonds":
        if "rating" in df.columns:
            bad = sorted(set(df["rating"].dropna().astype(str)) - RATINGS)
            if bad: warnings.append(f"Unexpected rating values (sample): {bad[:5]}")
        if "sector" in df.columns:
            bad = sorted(set(df["sector"].dropna().astype(str)) - SECTORS)
            if bad: warnings.append(f"Unexpected sector values (sample): {bad[:5]}")
        if "day_count" in df.columns:
            bad = sorted(set(df["day_count"].dropna().astype(str)) - DAY_COUNTS)
            if bad: warnings.append(f"Unexpected day_count values (sample): {bad[:5]}")

    report["errors"] = _limited(errors)
    report["warnings"] = _limited(warnings)
    report["passed"] = len(errors) == 0
    return report


# ------------------------
# Cross-table sanity checks
# ------------------------
def _cross_checks(bonds: pd.DataFrame, trades: pd.DataFrame) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    # 1) Referential integrity
    bad_isins = np.setdiff1d(trades["isin"].dropna().unique(), bonds["isin"].dropna().unique())
    if len(bad_isins) > 0:
        errors.append(f"Trades reference unknown ISINs: {len(bad_isins)} (e.g. {', '.join(_limited(bad_isins.tolist(), 5))})")

    # 2) Bonds sanity
    if {"maturity", "issue_date"}.issubset(bonds.columns):
        wrong_order = int((pd.to_datetime(bonds["maturity"]) <= pd.to_datetime(bonds["issue_date"])).sum())
        if wrong_order > 0:
            errors.append(f"bonds: maturity <= issue_date rows: {wrong_order}")

    if "coupon" in bonds.columns:
        out = bonds.loc[~bonds["coupon"].between(0.0, 12.0, inclusive="both")]
        if len(out) > 0:
            warnings.append(f"bonds: coupon outside [0,12]%: {len(out)}")

    if "oas0_bps" in bonds.columns:
        out = bonds.loc[~bonds["oas0_bps"].between(-50, 600)]
        if len(out) > 0:
            warnings.append(f"bonds: oas0_bps outside [-50,600]: {len(out)}")

    if "dv01_per_100" in bonds.columns:
        zeros = int((bonds["dv01_per_100"] <= 0).sum())
        if zeros > 0:
            errors.append(f"bonds: non-positive dv01_per_100 rows: {zeros}")

    # 3) Trades basic values
    neg_sizes = int((trades["size"] <= 0).sum()) if "size" in trades.columns else 0
    if neg_sizes > 0:
        errors.append(f"trades: non-positive size rows: {neg_sizes}")

    if "price" in trades.columns:
        nonpos_prices = int((trades["price"] <= 0).sum())
        if nonpos_prices > 0:
            errors.append(f"trades: non-positive clean price rows: {nonpos_prices}")

    if "side" in trades.columns:
        bad_sides = sorted(set(trades["side"].dropna()) - {"BUY","SELL"})
        if bad_sides:
            errors.append(f"trades: unexpected side values: {bad_sides}")

    # 4) Temporal logic
    if {"exec_time","report_time"}.issubset(trades.columns):
        bad = int((pd.to_datetime(trades["exec_time"]) > pd.to_datetime(trades["report_time"])).sum())
        if bad > 0:
            warnings.append(f"trades: exec_time > report_time rows: {bad}")

    # 5) TRACE-ish arithmetic & flags
    # reported_size <= size; cap indicator matches (size > cap_threshold)
    if {"reported_size","size"}.issubset(trades.columns):
        bad = int((trades["reported_size"] > trades["size"]).sum())
        if bad > 0:
            errors.append(f"trades: reported_size > size rows: {bad}")

    if {"trace_cap_indicator","size","cap_threshold"}.issubset(trades.columns):
        mask = trades["size"] > trades["cap_threshold"]
        mismatch = int((trades["trace_cap_indicator"].astype(bool) != mask.astype(bool)).sum())
        if mismatch > 0:
            errors.append(f"trades: trace_cap_indicator mismatch rows: {mismatch}")

    # price_dirty_exec ≈ price_clean_exec + accrued_interest
    if {"price_dirty_exec","price_clean_exec","accrued_interest"}.issubset(trades.columns):
        diff = (trades["price_dirty_exec"] - (trades["price_clean_exec"] + trades["accrued_interest"])).abs()
        bad = int((diff > 1e-6).sum())
        if bad > 0:
            warnings.append(f"trades: dirty != clean+accrued (>{1e-6:g}) rows: {bad}")

        neg_acc = int((trades["accrued_interest"] < 0).sum())
        if neg_acc > 0:
            errors.append(f"trades: negative accrued_interest rows: {neg_acc}")

    # dv01_dollar == dv01_per_100 * (size/100)
    if {"dv01_dollar","dv01_per_100","size"}.issubset(trades.columns):
        diff = (trades["dv01_dollar"] - trades["dv01_per_100"] * (trades["size"] / 100.0)).abs()
        bad = int((diff > 1e-6).sum())
        if bad > 0:
            warnings.append(f"trades: dv01_dollar formula mismatch rows: {bad}")

    # decomposition: y_bps ≈ y_pi_ref_bps + y_h_bps + y_delta_port_bps + y_eps_bps
    comp_cols = {"y_bps","y_pi_ref_bps","y_h_bps","y_delta_port_bps","y_eps_bps"}
    if comp_cols.issubset(trades.columns):
        recon = (trades["y_pi_ref_bps"] + trades["y_h_bps"] + trades["y_delta_port_bps"] + trades["y_eps_bps"])
        diff = (trades["y_bps"] - recon).abs()
        bad = int((diff > 1e-5).sum())
        if bad > 0:
            warnings.append(f"trades: y_bps decomposition mismatch (>{1e-5:g}) rows: {bad}")

    # portfolio invariants
    if {"sale_condition4","is_portfolio"}.issubset(trades.columns):
        port_flag = trades["sale_condition4"].fillna("") == "P"
        mismatch = int((port_flag != trades["is_portfolio"].astype(bool)).sum())
        if mismatch > 0:
            errors.append(f"trades: sale_condition4='P' ⇔ is_portfolio mismatch rows: {mismatch}")

    if {"is_portfolio","portfolio_id"}.issubset(trades.columns):
        missing_id = int(trades.loc[trades["is_portfolio"].astype(bool) & trades["portfolio_id"].isna()].shape[0])
        if missing_id > 0:
            errors.append(f"trades: portfolio rows missing portfolio_id: {missing_id}")

    if {"is_portfolio","y_delta_port_bps"}.issubset(trades.columns):
        zeros = int(trades.loc[trades["is_portfolio"].astype(bool) & (trades["y_delta_port_bps"] == 0)].shape[0])
        if zeros > 0:
            warnings.append(f"trades: portfolio lines with zero y_delta_port_bps: {zeros}")

    # yield scale sanity: decimal yields usually within (-5%, 50%) in any synthetic or real set
    if "yield_exec" in trades.columns:
        out = trades.loc[~trades["yield_exec"].between(-0.05, 0.50)]
        if len(out) > 0:
            warnings.append(f"trades: yield_exec outside [-5%, 50%] decimal: {len(out)}")

    # multiples of 5k size (typical TRACE rounding for corp bonds)
    if "size" in trades.columns:
        multiples = int((trades["size"] % 5000 != 0).sum())
        if multiples > 0:
            warnings.append(f"trades: sizes not multiple of 5,000: {multiples}")

    return {
        "errors": _limited(errors),
        "warnings": _limited(warnings),
        "passed": len(errors) == 0
    }


# -------------------------
# Public validation function
# -------------------------
def validate_raw(rawdir: Path) -> Dict[str, Any]:
    """
    Validate the raw schema: bonds.parquet + trades.parquet.
    Checks:
      - schema & dtypes (errors)
      - key uniqueness and nulls (errors)
      - referential integrity trades→bonds (errors)
      - value ranges & arithmetic identities (warnings/errors)
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
                "issue_date": "datetime",
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

    cross_report = {"errors": [], "warnings": [], "passed": False}
    if "bonds" in name_to_df and "trades" in name_to_df:
        cross_report = _cross_checks(name_to_df["bonds"], name_to_df["trades"])

    passed_tables = all(tr.get("passed", False) for tr in table_reports)
    passed_cross = cross_report.get("passed", False)
    return {
        "rawdir": str(rawdir),
        "passed": bool(passed_tables and passed_cross),
        "tables": table_reports,
        "cross_checks": cross_report,
    }
