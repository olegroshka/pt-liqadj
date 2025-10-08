from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from pathlib import Path
import logging
import numpy as np
import pandas as pd

from ptliq.utils.randomness import set_seed

@dataclass
class SimParams:
    n_bonds: int
    n_days: int
    providers: list[str]
    seed: int
    outdir: Path

RATINGS = ["AAA","AA","A","BBB","BB","B"]
SECTORS = ["FIN","IND","UTIL","TEL","TECH"]

def _gen_bonds(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    coupons = rng.normal(3.0, 1.0, size=n).clip(0.0, 8.0)
    issue_years = rng.integers(2005, 2024, size=n)
    matur_years = issue_years + rng.integers(2, 15, size=n)
    df = pd.DataFrame({
        "isin": [f"SIM{idx:010d}" for idx in range(n)],
        "issuer": [f"ISS{rng.integers(1, 300):04d}" for _ in range(n)],
        "sector": rng.choice(SECTORS, size=n).tolist(),
        "rating": rng.choice(RATINGS, size=n, p=[0.02,0.08,0.25,0.35,0.2,0.1]).tolist(),
        "issue_date": [date(int(y), int(rng.integers(1,13)), int(rng.integers(1,28))) for y in issue_years],
        "maturity": [date(int(y), int(rng.integers(1,13)), int(rng.integers(1,28))) for y in matur_years],
        "coupon": coupons,
        "amount_out": rng.lognormal(mean=8.0, sigma=0.7, size=n),
        "curve_bucket": rng.choice(["2Y","5Y","10Y","30Y"], size=n).tolist(),
    })
    # NEW: ensure parquet reads them back as datetime64[ns]
    df["issue_date"] = pd.to_datetime(df["issue_date"])
    df["maturity"]   = pd.to_datetime(df["maturity"])
    return df

def _gen_trades(bonds: pd.DataFrame, n_days: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    start = datetime(2025, 1, 2, 9, 0, 0)
    rows = []
    for d in range(n_days):
        day = start + timedelta(days=int(d))
        # per day, sample ~5% of bonds
        idx = rng.choice(bonds.index.values, size=max(1, len(bonds)//20), replace=False)
        for i in idx:
            b = bonds.loc[i]
            mid = 100 + rng.normal(0, 0.5)
            side = rng.choice(["BUY","SELL"])
            size = float(rng.lognormal(mean=5.0, sigma=0.6))
            # simple skew: SELL below mid, BUY above mid
            price = float(mid + (0.02 if side=="BUY" else -0.02) + rng.normal(0, 0.05))
            rows.append({
                "ts": day + timedelta(minutes=int(rng.integers(0, 6*60))),
                "isin": b["isin"],
                "side": side,
                "size": size,
                "price": price,
                "is_portfolio": False
            })
    return pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)

def simulate(params: SimParams) -> dict[str, pd.DataFrame]:
    set_seed(params.seed)
    logging.getLogger(__name__).info("Simulating %d bonds over %d days", params.n_bonds, params.n_days)
    bonds = _gen_bonds(params.n_bonds, params.seed)
    trades = _gen_trades(bonds, params.n_days, params.seed)
    return {"bonds": bonds, "trades": trades}
