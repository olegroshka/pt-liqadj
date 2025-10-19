from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ptliq.data.validate import validate_raw

def test_validator_catches_bad_values(tmp_path: Path):
    # bonds with bad maturity ordering
    bonds = pd.DataFrame({
        "isin": ["X1","X2"],
        "issuer": ["I1","I2"],
        "sector": ["FIN","IND"],
        "rating": ["A","BBB"],
        "issue_date": pd.to_datetime(["2025-01-10","2025-01-10"]),
        "maturity":   pd.to_datetime(["2025-01-01","2026-01-01"]),  # first is invalid (before issue)
        "coupon": [3.0, 4.0],
        "amount_out": [1e7, 2e7],
        "curve_bucket": ["5Y","10Y"],
    })
    now = pd.to_datetime("2025-01-15 10:00:00")
    trades = pd.DataFrame({
        "ts": [now, now + timedelta(minutes=5)],
        "isin": ["X1","BADISIN"],  # second invalid ISIN
        "side": ["BUY","HOLD"],    # invalid side
        "size": [1_000_000, -5.0], # negative size
        "price": [100.1, -1.0],    # non-positive price
        "is_portfolio": [False, False],
    })
    bonds.to_parquet(tmp_path / "bonds.parquet", index=False)
    trades.to_parquet(tmp_path / "trades.parquet", index=False)

    report = validate_raw(tmp_path)
    assert not report["passed"]
    # at least some of the expected failures appear
    cx = report.get("cross_checks")
    if isinstance(cx, dict):
        flat = (cx.get("errors", []) or []) + (cx.get("warnings", []) or [])
    else:
        flat = cx or []
    errs = " ".join([str(s) for s in flat])
    assert "unknown isin" in errs.lower()
    assert "non-positive" in errs.lower() or "unexpected side" in errs.lower()
