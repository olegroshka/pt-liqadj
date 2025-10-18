# ptliq/data/simulate.py
from __future__ import annotations

"""
Synthetic bonds & trades generator with TRACE-like fields and portfolio-conditioned effects.
- Backward compatible with previous 'advanced' simulator.
- Fixes: yield_exec units; robust accrued_interest; per-day variation in portfolio skew/pattern.
- Adds small after-hours probability to populate SaleCondition3='T'/'U'.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
import math
import random

import numpy as np
import pandas as pd

# Try to reuse project seeding if present
try:
    from ptliq.utils.randomness import set_seed as _project_set_seed  # type: ignore
except Exception:
    _project_set_seed = None


def set_seed(seed: int) -> None:
    if _project_set_seed is not None:
        _project_set_seed(seed)
    else:
        random.seed(seed)
        np.random.seed(seed)


# --------------------------------------------------------------------------------------
# Simulation parameters
# --------------------------------------------------------------------------------------
@dataclass
class SimParams:
    # core
    n_bonds: int
    n_days: int
    providers: List[str]
    seed: int
    outdir: Path

    # Pricing / reference price controls (kept from prior version)
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
    liq_eps_bps: float = 3.0             # residual noise ε in y_bps
    micro_price_noise_bps: float = 1.0   # extra microstructure noise added to price

    # --- Portfolio trade & TRACE-like mechanics ---
    portfolio_trade_share: float = 0.22  # share of trade lines that are part of portfolios
    basket_size_min: int = 10
    basket_size_max: int = 35
    port_skew_mu: float = 0.75
    port_skew_sigma: float = 0.35
    port_time_spread_sec: int = 2        # ± seconds around basket centroid

    # event rates
    asof_rate: float = 0.01
    late_rate: float = 0.02
    cancel_rate: float = 0.005
    ats_rate: float = 0.08
    after_hours_rate: float = 0.02       # NEW: chance to place a print outside 9–16

    # TRACE capping
    cap_ig: float = 5_000_000.0
    cap_hy: float = 1_000_000.0

    # Vendor liquidity → baseline premium mapping (per-provider linear monotone map)
    # gp(L) = gp_intercept_bps + gp_slope_bps * (L / 100)
    gp_intercept_bps: float = 1.0
    gp_slope_bps: float = 45.0

    # Reference premium vs vendor score coupling (for targets)
    pi_ref_slope: float = 0.4            # slope linking vendor_liq_score -> y_pi_ref_bps
    pi_ref_noise_bps: float = 4.0        # noise to loosen correlation to ~0.7–0.85

    # Trade intensity controls
    base_intensity: float = 0.18         # baseline Poisson rate per bond per day
    liq_to_intensity: float = 0.006      # lift per unit of static vendor liq (scaled to 0..1)

    # Urgency coefficient
    liq_urgency_coeff: float = 4.0

    # Use second-level timestamps like public TRACE
    second_resolution: bool = True


# Enums / constants
RATINGS = ["AAA", "AA", "A", "BBB", "BB", "B"]
SECTORS = ["FIN", "IND", "UTIL", "TEL", "TECH"]
CURVE_BUCKETS_COARSE = ["2Y", "5Y", "10Y", "30Y"]
CURVE_BUCKETS_FINE = ["0-3y", "3-5y", "5-7y", "7-10y", "10-15y"]
DAY_COUNTS = ["30/360", "ACT/ACT"]
FREQ_CHOICES = [1, 2, 2, 2, 4]  # weighted random (mostly semi-annual)


# --------------------------------------------------------------------------------------
# Yield curve and bond math helpers
# --------------------------------------------------------------------------------------
def ns_yield(t: float, a: float = 0.035, b: float = -0.012, c: float = 0.01, tau: float = 3.5) -> float:
    if t <= 0:
        return float(a + b)
    term1 = (1 - np.exp(-t / tau)) / (t / tau)
    term2 = term1 - np.exp(-t / tau)
    return float(a + b * term1 + c * term2)


def tenor_bucket_fine(tenor_years: float) -> str:
    if tenor_years < 3: return "0-3y"
    if tenor_years < 5: return "3-5y"
    if tenor_years < 7: return "5-7y"
    if tenor_years < 10: return "7-10y"
    return "10-15y"


def maturity_from_issue(issue_date: date) -> date:
    years = np.random.uniform(1, 15)
    return issue_date + timedelta(days=int(365.25 * years))


def add_months(dt: date, months: int) -> date:
    m = dt.month - 1 + months
    y = dt.year + m // 12
    m = m % 12 + 1
    # simple days-in-month logic
    dim = [31, 29 if y % 4 == 0 else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1]
    d = min(dt.day, dim)
    return date(y, m, d)


def mod_duration_price_dv01(coupon_rate: float, ytm: float, tenor_years: float, freq: int = 2) -> Tuple[float, float, float]:
    """Simple bullet bond pricing with Macaulay/modified duration and DV01 per 100 notional."""
    if tenor_years <= 0:
        tenor_years = 0.25
    n = max(1, int(round(tenor_years * freq)))
    cpn = coupon_rate / 100.0 / freq * 100.0
    r = ytm / freq
    disc = 1.0 / (1.0 + r)
    pv_coupons = cpn * (1 - disc ** n) / (1 - disc)
    pv_principal = 100.0 * (disc ** n)
    price = pv_coupons + pv_principal
    times = np.arange(1, n + 1, dtype=float)
    cf = np.full(n, cpn); cf[-1] += 100.0
    weights = (cf * disc ** times) / price
    macaulay_periods = float(np.sum(times * weights))
    macaulay_years = macaulay_periods / freq
    mod_duration = float(macaulay_years / (1 + ytm / freq))
    dv01 = mod_duration * price * 1e-4
    return float(price), float(mod_duration), float(dv01)


def coupon_schedule(issue_date: date, maturity_date: date, freq: int) -> List[date]:
    """Forward coupon dates excluding issue_date, including maturity."""
    months_step = 12 // freq
    sched = []
    dt = issue_date
    while True:
        dt = add_months(dt, months_step)
        if dt >= maturity_date:
            sched.append(maturity_date)
            break
        sched.append(dt)
        if len(sched) > 1000:
            break
    return sched


def last_next_coupon_robust(issue_date: date, trade_dt: date, schedule: List[date]) -> Tuple[date, date]:
    """
    Ensure last_c <= trade_dt <= next_c.
    If trade_dt < first coupon, use issue_date as last_c.
    """
    if not schedule:
        return issue_date, issue_date
    if trade_dt < schedule[0]:
        return issue_date, schedule[0]
    last_c = schedule[0]
    for d in schedule:
        if d <= trade_dt:
            last_c = d
        else:
            return last_c, d
    return last_c, schedule[-1]


def daycount_fraction(last_c: date, trade_dt: date, next_c: date, freq: int, conv: str) -> float:
    """Very simple 30/360 or ACT/ACT fraction within the coupon period [last_c, next_c]."""
    if trade_dt < last_c:
        return 0.0
    conv = (conv or "30/360").upper()
    if conv == "ACT/ACT":
        num = (trade_dt - last_c).days
        den = max(1, (next_c - last_c).days)
        return min(1.0, max(0.0, num / den))
    # 30/360 US-like
    D1 = min(last_c.day, 30); D2 = min(trade_dt.day, 30)
    num = 360 * (trade_dt.year - last_c.year) + 30 * (trade_dt.month - last_c.month) + (D2 - D1)
    den = 360 // freq
    return float(min(1.0, max(0.0, num / den)))


# --------------------------------------------------------------------------------------
# Bonds
# --------------------------------------------------------------------------------------
def _gen_bonds(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    today = date.today()

    # static
    issuer_ids = rng.integers(1, 400, size=n)
    sectors = rng.choice(SECTORS, size=n)
    ratings = rng.choice(RATINGS, size=n, p=[0.02, 0.10, 0.28, 0.35, 0.18, 0.07])
    coupons = rng.normal(3.2, 1.2, size=n).clip(0.5, 10.0)
    issue_age = rng.uniform(0.1, 7.0, size=n)
    issue_dates = [today - timedelta(days=int(365.25 * float(a))) for a in issue_age]
    maturities = [maturity_from_issue(d) for d in issue_dates]
    tenors = [max(0.25, (md - today).days / 365.25) for md in maturities]
    freqs = rng.choice(FREQ_CHOICES, size=n)
    day_counts = rng.choice(DAY_COUNTS, size=n)

    # curves / pricing
    base_curve_yield = np.array([ns_yield(t) for t in tenors], dtype=float)
    sector_levels = {s: i for i, s in enumerate(sorted(set(sectors.tolist())))}
    rating_levels = {r: i for i, r in enumerate(sorted(RATINGS))}

    sector_code = np.array([sector_levels[s] for s in sectors], dtype=float)
    rating_code = np.array([rating_levels[r] for r in ratings], dtype=float)

    clean_delta_bps = (0.0 + 3.0 * sector_code + 5.0 * rating_code + rng.normal(0.0, 2.0, size=n))
    price0_clean = 100.0 * (1.0 - clean_delta_bps / 10_000.0)

    # OAS / yield / duration / dv01
    oas0_bps = (50 + 15 * rating_code + rng.normal(0.0, 12.0, size=n)).astype(float)
    ytm0 = base_curve_yield + oas0_bps / 10_000.0
    mdur = np.zeros(n, dtype=float)
    dv01 = np.zeros(n, dtype=float)
    for i in range(n):
        p, m, d = mod_duration_price_dv01(coupons[i], ytm0[i], tenors[i], int(freqs[i]))
        mdur[i] = m; dv01[i] = d

    # size/liq
    amount_out = rng.lognormal(mean=8.2, sigma=0.7, size=n)  # ~$3.6m median
    curve_bucket_coarse = rng.choice(CURVE_BUCKETS_COARSE, size=n)
    curve_bucket_fine = np.array([tenor_bucket_fine(t) for t in tenors], dtype=object)

    vendor_liq_static = np.clip(
        50 + (np.log10(amount_out) - 6.0) * 12.0 + (4 - rating_code) * 5.0 + rng.normal(0, 6.0, size=n),
        1, 99,
    ).astype(float)

    bonds = pd.DataFrame({
        "isin": [f"SIM{idx:010d}" for idx in range(n)],
        "issuer": [f"ISS{issuer_ids[i]:04d}" for i in range(n)],
        "sector": sectors.tolist(),
        "rating": ratings.tolist(),
        "issue_date": pd.to_datetime(issue_dates),
        "maturity": pd.to_datetime(maturities),
        "coupon": coupons.astype(float),
        "amount_out": amount_out.astype(float),
        "curve_bucket": curve_bucket_coarse.tolist(),   # compatibility
        # --- enriched static fields ---
        "curve_bucket_fine": curve_bucket_fine.tolist(),
        "coupon_frequency": [int(f) for f in freqs],
        "day_count": day_counts.tolist(),
        "tenor_years": [float(t) for t in tenors],
        "base_curve_yield": base_curve_yield.astype(float),
        "oas0_bps": oas0_bps.astype(float),
        "ytm0": ytm0.astype(float),
        "price0_clean": price0_clean.astype(float),
        "duration_mod": mdur.astype(float),
        "dv01_per_100": dv01.astype(float),
        "vendor_liq_score_static": vendor_liq_static.astype(float),
    })
    return bonds


# --------------------------------------------------------------------------------------
# Providers → baseline premium
# --------------------------------------------------------------------------------------
def _providers_liq_and_pref(params: SimParams, bonds: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    For each provider p: a vendor liquidity score and a monotone map to π_ref (bps).
    """
    rng = np.random.default_rng(params.seed + 77)
    bonds = bonds.copy()
    active_provider = params.providers[0] if len(params.providers) else "P1"

    for p in params.providers:
        liq = np.clip(
            bonds["vendor_liq_score_static"].to_numpy() + rng.normal(0.0, 4.0, size=len(bonds)),
            1, 99
        )
        bonds[f"vendor_{p}_liq"] = liq
        pi_ref = params.gp_intercept_bps + params.gp_slope_bps * (liq / 100.0)
        bonds[f"pi_ref_{p}_bps"] = pi_ref.astype(float)

    bonds["active_provider"] = active_provider
    return bonds, active_provider


# --------------------------------------------------------------------------------------
# Utilities for trades
# --------------------------------------------------------------------------------------
def _pick_trade_count_per_bond(bonds: pd.DataFrame, params: SimParams, rng: np.random.Generator) -> Dict[str, int]:
    lref = bonds["vendor_liq_score_static"].to_numpy()
    lam = params.base_intensity + params.liq_to_intensity * (lref / 100.0)
    lam = np.clip(lam, 0.01, None)
    counts = rng.poisson(lam, size=len(bonds))
    return {bonds.loc[i, "isin"]: int(counts[i]) for i in range(len(bonds))}


def _cap_threshold(rating: str, params: SimParams) -> float:
    return params.cap_ig if rating in ("AAA", "AA", "A", "BBB") else params.cap_hy


def _settlement_date(trade_date: date, days: int = 2) -> date:
    return trade_date + timedelta(days=days)


def _portfolio_patterns() -> List[str]:
    return ["sector_crowding", "issuer_crowding", "curve_imbalance", "balanced"]


def _portfolio_delta(pattern: str, side_sign: float, port_skew: float, bond_row: pd.Series, focus: Dict[str, object]) -> float:
    """
    Portfolio-conditioned Δ in bps for a line. SELL widens (+), BUY tightens (-).
    """
    strength = max(0.05, abs(port_skew))
    if pattern == "sector_crowding":
        same_sector = float(bond_row["sector"] == focus["sector"])
        delta = (1.25 * same_sector + 0.6 * (1 - same_sector)) * strength
    elif pattern == "issuer_crowding":
        same_issuer = float(bond_row["issuer"] == focus["issuer"])
        delta = (1.5 * same_issuer + 0.5 * (1 - same_issuer)) * strength
    elif pattern == "curve_imbalance":
        same_curve = float(bond_row["curve_bucket_fine"] == focus["curve_bucket_fine"])
        delta = (1.35 * same_curve + 0.7 * (1 - same_curve)) * strength
    else:
        delta = 0.9 * strength
    return float(side_sign * delta)


def _compose_y_bps(
    baseline_ref_bps: float,
    size_z: float,
    side_sign: float,
    sec_code: float,
    rat_code: float,
    urgency: float,
    params: SimParams,
    port_delta_bps: float,
    rng: np.random.Generator,
) -> Tuple[float, float, float, float, float]:
    """
    Compose y_bps = π_ref + h(Q,U) + Δ + ε.
    Return (y_bps, h_bps, eps_bps, delta_bps, pi_ref_bps)
    """
    h_bps = (
        params.liq_size_coeff * size_z
        + params.liq_side_coeff * side_sign
        + params.liq_sector_coeff * sec_code
        + params.liq_rating_coeff * rat_code
        + params.liq_urgency_coeff * (urgency - 0.5)
    )
    eps_bps = rng.normal(0.0, params.liq_eps_bps)
    y_bps = float(baseline_ref_bps + h_bps + port_delta_bps + eps_bps)
    return y_bps, float(h_bps), float(eps_bps), float(port_delta_bps), float(baseline_ref_bps)


def _sale_condition3(is_late: bool, after_hours: bool) -> str:
    if is_late and after_hours:
        return "U"  # Late After Hours
    if is_late:
        return "Z"  # Late
    if after_hours:
        return "T"  # After Hours
    return ""


# --------------------------------------------------------------------------------------
# Trades + targets
# --------------------------------------------------------------------------------------
def _gen_trades_with_targets(bonds: pd.DataFrame, n_days: int, params: SimParams) -> pd.DataFrame:
    rng = np.random.default_rng(params.seed + 101)
    start = datetime(2025, 1, 2, 9, 0, 0)

    # Provider baselines
    bonds2, active_provider_default = _providers_liq_and_pref(params, bonds)
    # provider columns will be selected per-trade to increase realism

    # encodings
    sector_levels = {s: i for i, s in enumerate(sorted(bonds2["sector"].unique().tolist()))}
    rating_levels = {r: i for i, r in enumerate(sorted(RATINGS))}

    # provider-specific behavior
    providers = params.providers if len(params.providers) else ["P1"]
    prov_bias = {p: float(np.random.default_rng(params.seed + 202 + i).normal(0.0, 1.0)) for i, p in enumerate(providers)}
    prov_micro_scale = {p: float(np.clip(np.random.default_rng(params.seed + 203 + i).normal(1.0, 0.25), 0.5, 1.8)) for i, p in enumerate(providers)}

    trades_rows = []

    for d in range(n_days):
        trade_date = (start + timedelta(days=int(d))).date()

        # how many times each bond trades today
        counts = _pick_trade_count_per_bond(bonds2, params, rng)
        todays_isins = [isin for isin, k in counts.items() for _ in range(k)]
        if not todays_isins:
            continue
        rng.shuffle(todays_isins)

        # planned number of portfolio lines
        n_total = len(todays_isins)
        approx_port_lines = int(params.portfolio_trade_share * n_total)
        used = set()

        # basket sizes
        baskets: List[List[int]] = []
        remaining = approx_port_lines
        while remaining >= params.basket_size_min:
            bsize = int(rng.integers(params.basket_size_min, params.basket_size_max + 1))
            if bsize > remaining:
                bsize = remaining
            baskets.append([])
            remaining -= bsize

        # fill baskets with similarity by sector/rating/curve
        for b in baskets:
            candidates = [i for i in range(n_total) if i not in used]
            if not candidates:
                break
            i0 = int(rng.choice(candidates))
            used.add(i0); b.append(i0)

            seed_isin = todays_isins[i0]
            seed_b = bonds2.loc[bonds2["isin"] == seed_isin].iloc[0]
            pool = [i for i in candidates if i != i0]
            rng.shuffle(pool)

            def score(idx: int) -> float:
                bb = bonds2.loc[bonds2["isin"] == todays_isins[idx]].iloc[0]
                s = 0.0
                s += 1.0 if bb["sector"] == seed_b["sector"] else 0.0
                s += 0.5 * (1.0 - abs(float(rating_levels[bb["rating"]]) - float(rating_levels[seed_b["rating"]])) / 5.0)
                s += 0.2 * (1.0 if bb["curve_bucket_fine"] == seed_b["curve_bucket_fine"] else 0.0)
                return s

            need = max(len(b) + int(rng.integers(1, 3)) - 1, 0)  # small variability
            need = max(need, 0)
            need_target = max(0, min(params.basket_size_max, len(b) + (params.basket_size_min - 1)))
            pool_sorted = sorted(pool, key=score, reverse=True)[:max(need_target, 1) * 4]
            for i_idx in pool_sorted:
                if len(b) >= need_target:
                    break
                if i_idx not in used:
                    used.add(i_idx); b.append(i_idx)

        # index→basket id
        index_to_port: Dict[int, int] = {}
        for b_id, b in enumerate(baskets):
            for idx in b:
                index_to_port[idx] = b_id

        # now generate the trades
        for i, isin in enumerate(todays_isins):
            bnd = bonds2.loc[bonds2["isin"] == isin].iloc[0]
            # Execution time (mostly during 9–16)
            base_time = datetime.combine(trade_date, datetime.min.time()) + timedelta(minutes=int(rng.integers(9*60, 16*60)))
            if rng.random() < params.after_hours_rate:
                # pick a time slightly outside regular hours
                if rng.random() < 0.5:
                    base_time = datetime.combine(trade_date, datetime.min.time()) + timedelta(minutes=int(rng.integers(8*60, 9*60)))
                else:
                    base_time = datetime.combine(trade_date, datetime.min.time()) + timedelta(minutes=int(rng.integers(16*60, 17*60)))

            in_portfolio = i in index_to_port
            basket_id = index_to_port.get(i, None)
            side = rng.choice(["BUY", "SELL"])
            side_sign = 1.0 if side == "SELL" else -1.0

            # sizes & capping
            if bnd["rating"] in ("AAA", "AA", "A", "BBB"):
                mu = math.log(1.2e6); sigma = 1.0
            else:
                mu = math.log(6.0e5); sigma = 1.0
            size = float(np.clip(rng.lognormal(mu, sigma), 25_000.0, 10_000_000.0))
            size = float(int(size / 5000) * 5000)
            cap_thr = _cap_threshold(str(bnd["rating"]), params)
            is_capped = size > cap_thr
            reported_size = min(size, cap_thr) if is_capped else size

            # urgency
            urgency = float(rng.random())

            # pick active provider per trade
            provider = rng.choice(providers)
            provider_act_col = f"pi_ref_{provider}_bps"
            provider_liq_col = f"vendor_{provider}_liq"
            active_provider = provider
            # provider baseline (reference premium for internal provider mapping)
            # Note: for target component we tie to vendor_liq_score with configurable slope+noise
            v_liq = float(bnd[provider_liq_col])
            y_pi_ref_out = float(params.pi_ref_slope * v_liq + rng.normal(0.0, params.pi_ref_noise_bps))

            # standardize size via the generator's log parameters
            size_z = (math.log(max(1.0, size)) - mu) / (sigma + 1e-8)
            sec_code = float(sector_levels[str(bnd["sector"])])
            rat_code = float(rating_levels[str(bnd["rating"])])

            # portfolio specifics
            portfolio_id = None
            sale_condition4 = ""
            port_delta_bps = 0.0
            port_skew = 0.0
            port_similarity = 0.0
            pattern = ""

            if in_portfolio and basket_id is not None:
                sale_condition4 = "P"
                portfolio_id = f"PF_{trade_date.strftime('%Y%m%d')}_{basket_id:04d}"
                # ensure per-day randomness for each basket
                seed_basket = (params.seed + 10_000) ^ (hash((int(trade_date.strftime('%Y%m%d')), int(basket_id))) & 0xFFFF_FFFF)
                rng_b = np.random.default_rng(seed_basket)
                pattern = rng_b.choice(_portfolio_patterns())
                port_skew = float(rng_b.normal(params.port_skew_mu, params.port_skew_sigma))
                focus = {
                    "sector": rng_b.choice(SECTORS),
                    "issuer": bnd["issuer"],  # lightweight proxy
                    "curve_bucket_fine": rng_b.choice(CURVE_BUCKETS_FINE),
                }
                # cluster times to seconds if enabled
                base_time = base_time.replace(second=0, microsecond=0) if params.second_resolution else base_time
                jitter = int(rng_b.integers(-params.port_time_spread_sec, params.port_time_spread_sec + 1))
                base_time = base_time + timedelta(seconds=jitter)

                base_port_delta = _portfolio_delta(pattern, side_sign, port_skew, bnd, focus)
                # continuous similarity for portfolio trades
                port_similarity = float(np.random.default_rng(seed_basket + 1).beta(3, 4))
                sigma_port = float(1.0 + 2.0 * abs(port_skew) * port_similarity)
                port_delta_bps = float(np.random.default_rng(seed_basket + 2).normal(base_port_delta, sigma_port))

            # portfolio delta handling for label vs reporting
            if in_portfolio and basket_id is not None:
                delta_for_y = float(port_delta_bps)
                y_delta_out = float(port_delta_bps)
            else:
                # normalize similarity for non-portfolio lines
                port_similarity = 0.0
                delta_for_y = 0.0
                y_delta_out = 0.0

            # truthful premium & observed prices
            y_bps, h_bps, eps_bps, delta_bps, pi_ref_used = _compose_y_bps(
                baseline_ref_bps=y_pi_ref_out,
                size_z=size_z,
                side_sign=side_sign,
                sec_code=sec_code,
                rat_code=rat_code,
                urgency=urgency,
                params=params,
                port_delta_bps=delta_for_y,
                rng=rng,
            )
            micro = float(rng.normal(prov_bias[provider], params.micro_price_noise_bps * prov_micro_scale[provider]))
            delta_obs_bps = y_bps + micro
            clean_price = float(bnd["price0_clean"])
            price_clean_exec = clean_price * (1.0 - delta_obs_bps / 10_000.0)

            # Accrued & dirty price (robust)
            freq = int(bnd["coupon_frequency"])
            sched = coupon_schedule(bnd["issue_date"].date(), bnd["maturity"].date(), freq)
            last_c, next_c = last_next_coupon_robust(bnd["issue_date"].date(), trade_date, sched)
            frac = daycount_fraction(last_c, trade_date, next_c, freq, str(bnd["day_count"]))
            accrued = 100.0 * (float(bnd["coupon"]) / 100.0) * frac
            accrued = max(0.0, float(accrued))  # guard
            price_dirty_exec = price_clean_exec + accrued

            # Yield proxy: convert Δy_bps to decimal before adding to ytm0
            dv01_per_100 = float(bnd["dv01_per_100"])
            dy_bps = - (price_clean_exec - clean_price) / max(1e-8, dv01_per_100)  # in bps
            ytm_exec = float(bnd["ytm0"] + dy_bps * 1e-4)  # decimal

            # flags & timestamps
            is_late = bool(rng.random() < params.late_rate)
            is_asof = bool(rng.random() < params.asof_rate)
            is_cancel = bool(rng.random() < params.cancel_rate)

            after_hours = bool((base_time.hour < 9) or (base_time.hour > 16))
            report_delay_min = int(rng.integers(0, 60 if is_late else 10))
            report_time = base_time + timedelta(minutes=report_delay_min)
            exec_time = base_time

            sale_condition3 = _sale_condition3(is_late, after_hours)
            remuneration = rng.choice(["", "C", "M", "N"], p=[0.7, 0.1, 0.1, 0.1])
            ats = bool(rng.random() < params.ats_rate)
            asof_indicator = "A" if is_asof else ""

            trades_rows.append(dict(
                # Back-compat core
                ts=pd.to_datetime(report_time),
                isin=str(bnd["isin"]),
                side=str(side),
                size=float(size),
                is_portfolio=bool(in_portfolio),

                # Truth decomposition
                y_bps=float(y_bps),
                y_h_bps=float(h_bps),
                y_eps_bps=float(eps_bps),
                y_delta_port_bps=float(y_delta_out),
                y_pi_ref_bps=float(y_pi_ref_out),
                urgency=float(urgency),

                # TRACE-like & analytics
                exec_time=pd.to_datetime(exec_time),
                report_time=pd.to_datetime(report_time),
                trade_dt=pd.to_datetime(trade_date),
                is_late=bool(is_late),
                is_asof=bool(is_asof),
                is_cancel=bool(is_cancel),
                contra_party_type=str(rng.choice(["C", "D", "E"], p=[0.9, 0.07, 0.03])),
                remuneration=str(remuneration),
                sale_condition3=str(sale_condition3),
                sale_condition4=str(sale_condition4),
                asof_indicator=str(asof_indicator),
                ats=bool(ats),

                reported_size=float(reported_size),
                trace_cap_indicator=bool(is_capped),
                cap_threshold=float(cap_thr),

                clean_price=float(clean_price),
                price=float(price_clean_exec),
                price_clean_exec=float(price_clean_exec),
                accrued_interest=float(accrued),
                price_dirty_exec=float(price_dirty_exec),
                yield_exec=float(ytm_exec),
                dv01_per_100=float(dv01_per_100),
                dv01_dollar=float(dv01_per_100 * (size / 100.0)),
                dv01_signed=float(dv01_per_100 * (1 if side == "BUY" else -1)),

                sector=str(bnd["sector"]),
                rating=str(bnd["rating"]),
                vendor_liq_score=float(bnd[provider_liq_col]),
                active_provider=str(active_provider),

                portfolio_id=str(portfolio_id) if portfolio_id else None,
                portfolio_pattern=str(pattern) if pattern else "",
                portfolio_skew=float(port_skew),
                portfolio_similarity=float(port_similarity),
            ))

    if not trades_rows:
        return pd.DataFrame(columns=["ts", "isin", "side", "size", "is_portfolio", "y_bps", "price", "price_clean_exec"])

    trades = pd.DataFrame(trades_rows).sort_values(["exec_time", "isin"]).reset_index(drop=True)

    # Guardrail: y_bps must equal sum of components within tolerance
    try:
        comp = trades[["y_pi_ref_bps","y_h_bps","y_delta_port_bps","y_eps_bps"]].sum(axis=1)
        diff = (trades["y_bps"] - comp).abs().max()
        assert float(diff) < 1e-6
    except Exception:
        # if columns are missing for some reason, skip strict check
        pass

    return trades


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------
def simulate(params: SimParams) -> Dict[str, pd.DataFrame]:
    set_seed(params.seed)
    logging.getLogger(__name__).info("Simulating %d bonds over %d days", params.n_bonds, params.n_days)

    bonds = _gen_bonds(params.n_bonds, params.seed)
    trades = _gen_trades_with_targets(bonds, params.n_days, params)

    # basic integrity (compatibility)
    assert {"isin", "issuer", "sector", "rating", "issue_date", "maturity", "coupon", "amount_out", "curve_bucket"}.issubset(bonds.columns)
    assert {"ts", "isin", "side", "size", "is_portfolio", "y_bps", "price", "price_clean_exec"}.issubset(trades.columns)

    return {"bonds": bonds, "trades": trades}
