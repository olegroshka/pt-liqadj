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

# ----------------------------
# Diagnostics helpers
# ----------------------------

def _missingness(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {}
    return {c: float(df[c].isna().mean()) for c in df.columns}


def _constant_like_columns(df: pd.DataFrame, threshold: float = 0.95) -> List[str]:
    out: List[str] = []
    if df.empty:
        return out
    n = len(df)
    for c in df.columns:
        vc = df[c].value_counts(dropna=False)
        top = int(vc.iloc[0]) if len(vc) > 0 else 0
        if n > 0 and (top / n) >= threshold:
            out.append(c)
    return out


def _near_duplicate_numeric(df: pd.DataFrame, tol: float = 1e-12) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    if df.empty:
        return pairs
    num = df.select_dtypes(include=["number"]).copy()
    cols = list(num.columns)
    for i in range(len(cols)):
        a = cols[i]
        for j in range(i + 1, len(cols)):
            b = cols[j]
            # compare only on rows where both are not null
            both = num[[a, b]].dropna()
            if both.empty:
                continue
            diffmax = float((both[a] - both[b]).abs().max())
            if diffmax <= tol:
                pairs.append((a, b))
    return pairs


def _top_correlations(df: pd.DataFrame, targets: List[str], k: int = 10) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    if df.empty:
        return out
    num = df.select_dtypes(include=["number"]).copy()
    if num.empty:
        return out
    corr = num.corr(method="pearson")
    for t in targets:
        if t in corr.columns:
            s = corr[t].drop(index=t).dropna().abs().sort_values(ascending=False).head(k)
            out[t] = {str(idx): float(val) for idx, val in s.items()}
    return out


# ---------------
# Table validators
# ---------------
def _check_table(path: Path, spec: TableSpec, val_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    report: Dict[str, Any] = {"table": spec.name, "path": str(path), "exists": path.exists()}
    if not path.exists():
        report["errors"] = [f"Missing table file: {path.name}"]
        report["warnings"] = []
        report["passed"] = False
        report["diagnostics"] = {}
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

    # diagnostics & soft checks
    miss = _missingness(df)
    const_cols = _constant_like_columns(df, threshold=0.95)
    near_dupes = _near_duplicate_numeric(df, tol=1e-12)

    # Apply validator config allow-lists
    def _extract_validation_section(cfg: Optional[Dict[str, Any] | Any]):
        if cfg is None:
            return {}
        # dict-like root
        if isinstance(cfg, dict):
            if "validation" in cfg:
                return cfg.get("validation", {})
            data = cfg.get("data") if isinstance(cfg.get("data"), dict) else None if not isinstance(cfg, dict) else cfg.get("data")
            if isinstance(data, dict) and "validation" in data:
                return data.get("validation", {})
            return {}
        # object-like root
        try:
            if hasattr(cfg, "validation") and getattr(cfg, "validation") is not None:
                return getattr(cfg, "validation")
            if hasattr(cfg, "data"):
                data = getattr(cfg, "data")
                # support both dict and object
                if isinstance(data, dict) and "validation" in data:
                    return data.get("validation", {})
                if hasattr(data, "validation"):
                    return getattr(data, "validation")
        except Exception:
            pass
        return {}

    val_cfg = _extract_validation_section(val_config)
    # low variation allow-list (suppress warnings for listed columns)
    low_var_allow_cfg = set(getattr(val_cfg, "low_variation_allowlist", []) if not isinstance(val_cfg, dict) else val_cfg.get("low_variation_allowlist", []))
    # expected-null rules: list of {column, when}
    expected_null_rules = (
        getattr(val_cfg, "expected_null", []) if not isinstance(val_cfg, dict) else val_cfg.get("expected_null", [])
    ) or (
        getattr(val_cfg, "expected_missing_by_condition", []) if not isinstance(val_cfg, dict) else val_cfg.get("expected_missing_by_condition", [])
    ) or []

    # Allow-list adjustments for trades table
    expected_missing_notes: List[str] = []
    if spec.name == "trades":
        # Suppress low-variation warnings for expected near-constants
        low_var_allow = {"is_late","is_asof","is_cancel","sale_condition3","asof_indicator","active_provider"} | low_var_allow_cfg
        const_cols = [c for c in const_cols if c not in low_var_allow]
        # Reclassify portfolio_id missingness as expected when is_portfolio == False
        if "portfolio_id" in df.columns and "is_portfolio" in df.columns:
            is_port = df["is_portfolio"].astype(bool)
            missing_mask = df["portfolio_id"].isna()
            missing_on_non_port = bool((missing_mask & ~is_port).all()) if len(df) > 0 else False
            missing_on_port = int((missing_mask & is_port).sum())
            if missing_on_non_port and missing_on_port == 0:
                # Remove from high-missing list later and record note
                expected_missing_notes.append("portfolio_id missing on non-portfolio rows (expected)")

    # warn on high missingness and constancy
    high_missing = [c for c, p in miss.items() if p > 0.02]

    # Config-driven expected-null removal from high-missing list
    try:
        for rule in expected_null_rules:
            col = str(rule.get("column", ""))
            when = str(rule.get("when", ""))
            if not col or col not in df.columns or not when:
                continue
            # build condition mask safely using pandas.eval
            try:
                cond_mask = df.eval(when)
                cond_mask = cond_mask.astype(bool)
            except Exception:
                # fallback: try query (will error if bad); if fails, skip rule
                try:
                    cond_mask = df.query(when).index.to_series().reindex(df.index, fill_value=False).astype(bool)
                except Exception:
                    continue
            missing_mask = df[col].isna()
            # If all missing values occur where condition is True, treat as expected
            if bool(((missing_mask) & (~cond_mask)).sum() == 0) and bool(missing_mask.any()):
                if col in high_missing:
                    high_missing = [c for c in high_missing if c != col]
                    expected_missing_notes.append(f"{col} missing under condition '{when}' (expected)")
    except Exception:
        pass

    # Remove portfolio_id from high-missing if expected by built-in rule
    if spec.name == "trades" and "portfolio_id" in high_missing and expected_missing_notes:
        high_missing = [c for c in high_missing if c != "portfolio_id"]
    if high_missing:
        warnings.append(f"high-missingness cols (>2%): {high_missing[:10]}")
    if const_cols:
        warnings.append(f"low-variation cols (>=95% same value): {const_cols[:10]}")
    if near_dupes:
        # allowlist of known compatibility aliases that are expected to be identical
        allow_dupes = set()
        if spec.name == "trades":
            if "price" in df.columns and "price_clean_exec" in df.columns:
                allow_dupes.add(("price", "price_clean_exec"))
                allow_dupes.add(("price_clean_exec", "price"))
        hard_dupes = [p for p in near_dupes if p not in allow_dupes]
        soft_dupes = [p for p in near_dupes if p in allow_dupes]
        if hard_dupes:
            errors.append(f"near-duplicate numeric columns (tol<=1e-12): {hard_dupes[:10]}")
        if soft_dupes:
            warnings.append(f"compatibility duplicate aliases (identical numeric cols): {soft_dupes[:10]}")

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
    # attach diagnostics for machine consumption
    report["diagnostics"] = {
        "missingness": miss,
        "constant_like_cols": const_cols,
        "near_duplicate_numeric_pairs": near_dupes,
        "nunique": {c: int(pd.Series(df[c]).nunique(dropna=True)) for c in df.columns},
    }
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
        # If capped, reported_size must equal cap_threshold
        if {"reported_size"}.issubset(trades.columns):
            capped = trades["trace_cap_indicator"].astype(bool)
            bad_eq = int((trades.loc[capped, "reported_size"] != trades.loc[capped, "cap_threshold"]).sum())
            if bad_eq > 0:
                errors.append(f"trades: capped rows with reported_size != cap_threshold: {bad_eq}")

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

    # portfolio delta policy: non-portfolio must be 0.0; no nulls anywhere
    if {"is_portfolio","y_delta_port_bps"}.issubset(trades.columns):
        nulls = int(trades["y_delta_port_bps"].isna().sum())
        if nulls > 0:
            errors.append(f"trades: y_delta_port_bps has nulls: {nulls}")
        non_port_bad = int((~trades["is_portfolio"].astype(bool) & (trades["y_delta_port_bps"].astype(float) != 0.0)).sum())
        if non_port_bad > 0:
            errors.append(f"trades: non-portfolio rows must have y_delta_port_bps == 0.0: {non_port_bad}")

    # Side effect on premiums: SELL should have higher mean y_bps than BUY
    if {"y_bps","side"}.issubset(trades.columns):
        try:
            grp = trades.groupby(trades["side"].astype(str))["y_bps"].mean()
            if {"SELL","BUY"}.issubset(set(grp.index)):
                if not (float(grp.get("SELL", 0.0)) > float(grp.get("BUY", 0.0))):
                    warnings.append("trades: expected mean(y_bps|SELL) > mean(y_bps|BUY) not observed")
        except Exception:
            pass

    # ATS share stability in [5%, 20%]
    if "ats" in trades.columns:
        try:
            p = float(trades["ats"].astype(bool).mean()) if len(trades) > 0 else float("nan")
            if np.isfinite(p) and not (0.05 <= p <= 0.20):
                warnings.append(f"trades: ats share {p:.3%} outside [5%, 20%]")
        except Exception:
            pass

    # leakage detector on numeric columns vs key targets
    leakage_diag: Dict[str, Dict[str, float]] = {}
    try:
        targets = [t for t in ["y_bps", "y_pi_ref_bps"] if t in trades.columns]
        leakage_diag = _top_correlations(trades, targets=targets, k=10)
        # flag extreme correlations
        for t, pairs in leakage_diag.items():
            for feat, rho in pairs.items():
                if not np.isfinite(rho):
                    continue
                if abs(rho) >= 0.99:
                    errors.append(f"leakage: |corr({feat},{t})| >= 0.99 → {rho:.3f}")
                elif abs(rho) >= 0.95:
                    warnings.append(f"leakage hint: |corr({feat},{t})| >= 0.95 → {rho:.3f}")
    except Exception:
        # keep validator robust if correlations fail
        pass

    return {
        "errors": _limited(errors),
        "warnings": _limited(warnings),
        "passed": len(errors) == 0,
        "diagnostics": {
            "leakage_top_correlations": leakage_diag,
        },
    }


# -------------------------
# Public validation function
# -------------------------
def validate_raw(rawdir: Path, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
        trep = _check_table(path, spec, val_config=config)
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
