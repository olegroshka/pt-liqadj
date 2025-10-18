from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from pathlib import Path
import logging

import numpy as np
import pandas as pd

from ptliq.utils.randomness import set_seed

# --------------------------------------------------------------------------------------
# Simulation parameters
# --------------------------------------------------------------------------------------
@dataclass
class SimParams:
    n_bonds: int
    n_days: int
    providers: list[str]
    seed: int
    outdir: Path
    # Pricing / reference price controls
    par: float = 100.0
    base_spread_bps: float = 0.0         # baseline deviation (clean price vs par)
    sector_spread_bps: float = 3.0       # sector tilt for clean price (per code step)
    rating_spread_bps: float = 5.0       # rating tilt for clean price (per code step)
    clean_price_noise_bps: float = 2.0   # idiosyncratic clean-price noise per bond
    # Liquidity premium generator (what the model learns as y_bps)
    liq_size_coeff: float = 8.0
    liq_side_coeff: float = 12.0         # SELL = +, BUY = -
    liq_sector_coeff: float = 2.0
    liq_rating_coeff: float = -2.0       # worse rating → larger positive premium
    liq_eps_bps: float = 3.0             # residual noise in y_bps
    # Micro price noise added on top of y_bps when forming trade price
    micro_price_noise_bps: float = 1.0

RATINGS = ["AAA", "AA", "A", "BBB", "BB", "B"]
SECTORS = ["FIN", "IND", "UTIL", "TEL", "TECH"]


# --------------------------------------------------------------------------------------
# Bonds
# --------------------------------------------------------------------------------------
def _gen_bonds(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    coupons = rng.normal(3.0, 1.0, size=n).clip(0.0, 8.0)
    issue_years = rng.integers(2005, 2024, size=n)
    matur_years = issue_years + rng.integers(2, 15, size=n)
    df = pd.DataFrame({
        "isin": [f"SIM{idx:010d}" for idx in range(n)],
        "issuer": [f"ISS{rng.integers(1, 300):04d}" for _ in range(n)],
        "sector": rng.choice(SECTORS, size=n).tolist(),
        "rating": rng.choice(RATINGS, size=n, p=[0.02, 0.08, 0.25, 0.35, 0.20, 0.10]).tolist(),
        "issue_date": [date(int(y), int(rng.integers(1, 13)), int(rng.integers(1, 28))) for y in issue_years],
        "maturity": [date(int(y), int(rng.integers(1, 13)), int(rng.integers(1, 28))) for y in matur_years],
        "coupon": coupons,
        "amount_out": rng.lognormal(mean=8.0, sigma=0.7, size=n),
        "curve_bucket": rng.choice(["2Y", "5Y", "10Y", "30Y"], size=n).tolist(),
    })
    # ensure parquet round-trips as datetime64
    df["issue_date"] = pd.to_datetime(df["issue_date"])
    df["maturity"] = pd.to_datetime(df["maturity"])
    return df


# --------------------------------------------------------------------------------------
# Trades (+ truthful y_bps) using bonds' clean_price reference
# --------------------------------------------------------------------------------------
def _gen_trades_with_targets(bonds: pd.DataFrame, n_days: int, params: SimParams) -> pd.DataFrame:
    rng = np.random.default_rng(params.seed + 1)
    start = datetime(2025, 1, 2, 9, 0, 0)

    # reference (clean) price per bond, driven by sector/rating codes + idiosyncratic noise
    sector_levels = sorted(bonds["sector"].unique().tolist())
    rating_levels = sorted(bonds["rating"].unique().tolist())
    sector_code_b = bonds["sector"].map({s: i for i, s in enumerate(sector_levels)}).to_numpy()
    rating_code_b = bonds["rating"].map({r: i for i, r in enumerate(rating_levels)}).to_numpy()
    clean_delta_bps = (
        params.base_spread_bps
        + params.sector_spread_bps * sector_code_b
        + params.rating_spread_bps * rating_code_b
        + rng.normal(0.0, params.clean_price_noise_bps, size=len(bonds))
    )
    bonds = bonds.copy()
    bonds["clean_price"] = params.par * (1.0 - clean_delta_bps / 10_000.0)

    # generate raw rows (no price yet)
    rows = []
    for d in range(n_days):
        day = start + timedelta(days=int(d))
        # ~5% of bonds trade each day
        idx = rng.choice(bonds.index.values, size=max(1, len(bonds) // 20), replace=False)
        for i in idx:
            b = bonds.loc[i]
            side = rng.choice(["BUY", "SELL"])
            size = float(rng.lognormal(mean=5.0, sigma=0.6))
            rows.append({
                "ts": day + timedelta(minutes=int(rng.integers(0, 6 * 60))),
                "isin": b["isin"],
                "side": side,
                "size": size,
                "is_portfolio": False,
            })

    trades = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)

    # join sector/rating/clean_price for feature/target generation (drop clean_price after)
    trades = trades.merge(
        bonds[["isin", "sector", "rating", "clean_price"]],
        on="isin", how="left", validate="m:1"
    )

    # side sign
    side_sign = trades["side"].astype(str).str.upper().map({"SELL": 1.0, "BUY": -1.0}).fillna(0.0).to_numpy()

    # size z-score within ISIN (simple local context)
    grp = trades.groupby("isin", observed=True)["size"]
    size_mean = grp.transform("mean")
    size_std = grp.transform("std").replace(0.0, 1.0)
    size_z = ((trades["size"] - size_mean) / size_std).to_numpy()

    # category codes (for y_bps generator)
    sec_code_t = trades["sector"].map({s: i for i, s in enumerate(sector_levels)}).to_numpy()
    rat_code_t = trades["rating"].map({r: i for i, r in enumerate(rating_levels)}).to_numpy()

    # truthful liquidity premium (what the model predicts): y_bps
    eps = rng.normal(0.0, params.liq_eps_bps, size=len(trades))
    y_bps = (
        params.liq_size_coeff * size_z
        + params.liq_side_coeff * side_sign
        + params.liq_sector_coeff * sec_code_t
        + params.liq_rating_coeff * rat_code_t
        + eps
    ).astype(np.float32)
    trades["y_bps"] = y_bps

    # observed price = clean_price * (1 - (y_bps + micro)/10k)
    micro = rng.normal(0.0, params.micro_price_noise_bps, size=len(trades))
    delta_obs_bps = y_bps + micro
    trades["price"] = trades["clean_price"] * (1.0 - delta_obs_bps / 10_000.0)

    # drop helper columns we don't want duplicated in parquet pipeline (keep sector/rating)
    trades = trades.drop(columns=["clean_price"])

    return trades


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------
def simulate(params: SimParams) -> dict[str, pd.DataFrame]:
    set_seed(params.seed)
    logging.getLogger(__name__).info(
        "Simulating %d bonds over %d days", params.n_bonds, params.n_days
    )

    bonds = _gen_bonds(params.n_bonds, params.seed)
    trades = _gen_trades_with_targets(bonds, params.n_days, params)

    return {"bonds": bonds, "trades": trades}
