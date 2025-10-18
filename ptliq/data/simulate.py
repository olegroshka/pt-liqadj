
from __future__ import annotations

"""
ptliq.data.simulate
-------------------
Synthetic data generator for bonds & trades with TRACE-like fields and
portfolio-conditioned price impacts. This merges the "current project" simulator
with the more advanced collaboration ideas and aligns to the hackathon PDF.

Design goals
============
1) Keep backward-compatibility with the prior API and columns:
   - bonds: ["isin","issuer","sector","rating","issue_date","maturity","coupon","amount_out","curve_bucket"]
   - trades: ["ts","isin","side","size","is_portfolio","y_bps","price"]
2) Add vendor-liquidity providers and a monotone mapping to a baseline premium π_ref (bps),
   plus an explicit portfolio delta Δ (bps) so that: y_bps = π_ref + h(Q,U) + Δ + ε.
3) Inject realistic portfolio-trade structure (SaleCondition4='P'), second-level timestamps,
   ATS/AsOf/Late flags, size capping, and TRACE-like fields needed later for reconstruction tests.
4) Expose clear, documented params to shape behaviors deterministically for backtests.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import List, Dict, Tuple, Optional
import logging
import math
import random

import numpy as np
import pandas as pd

# Try to use project seeding utility if present; otherwise define a local fallback.
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
    outdir: 'Path'

    # Pricing / reference price controls (kept from prior version)
    par: float = 100.0
    base_spread_bps: float = 0.0         # baseline deviation (clean price vs par)
    sector_spread_bps: float = 3.0       # sector tilt for clean price (per code step)
    rating_spread_bps: float = 5.0       # rating tilt for clean price (per code step)
    clean_price_noise_bps: float = 2.0   # idiosyncratic clean-price noise per bond

    # Liquidity premium generator (what the model learns as y_bps)
    # (naming kept to avoid breaking config/CLI; conceptually this is h(Q,U) + part of Δ)
    liq_size_coeff: float = 8.0
    liq_side_coeff: float = 12.0         # SELL = +, BUY = -
    liq_sector_coeff: float = 2.0
    liq_rating_coeff: float = -2.0       # worse rating → larger positive premium
    liq_eps_bps: float = 3.0             # residual noise ε in y_bps
    micro_price_noise_bps: float = 1.0   # extra microstructure noise added to price

    # --- New: portfolio trade & TRACE-like mechanics ---
    portfolio_trade_share: float = 0.22  # share of trade lines that are part of portfolios
    basket_size_min: int = 10            # FINRA definition threshold for portfolio list trades
    basket_size_max: int = 35
    port_skew_mu: float = 0.75           # average magnitude of Δ on portfolio lines
    port_skew_sigma: float = 0.35
    port_time_spread_sec: int = 2        # portfolio lines disseminated within ±N seconds

    # event rates
    asof_rate: float = 0.01
    late_rate: float = 0.02
    cancel_rate: float = 0.005
    ats_rate: float = 0.08

    # TRACE capping
    cap_ig: float = 5_000_000.0
    cap_hy: float = 1_000_000.0

    # Vendor liquidity → baseline premium calibration (per-provider linear monotone map)
    # gp(L) = gp_intercept_bps[p] + gp_slope_bps[p] * (L / 100)
    gp_intercept_bps: float = 1.0
    gp_slope_bps: float = 45.0

    # Trade intensity controls
    base_intensity: float = 0.18         # baseline Poisson rate per bond per day
    liq_to_intensity: float = 0.006      # how much static vendor_liq lifts intensity

    # Urgency coefficient (added to "h(Q,U)" term); kept optional
    liq_urgency_coeff: float = 4.0

    # Reproducibility: use second-level timestamps to mimic public TRACE precision
    second_resolution: bool = True


# Constants / enumerations
RATINGS = ["AAA", "AA", "A", "BBB", "BB", "B"]
SECTORS = ["FIN", "IND", "UTIL", "TEL", "TECH"]
CURVE_BUCKETS_COARSE = ["2Y", "5Y", "10Y", "30Y"]
CURVE_BUCKETS_FINE = ["0-3y", "3-5y", "5-7y", "7-10y", "10-15y"]
DAY_COUNTS = ["30/360", "ACT/ACT"]
FREQ_CHOICES = [1, 2, 2, 2, 4]  # weighted random (mostly semi-annual)


# --------------------------------------------------------------------------------------
# Helpers for bond math (simple but stable)
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
    d = min(dt.day, [31, 29 if y % 4 == 0 else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m-1])
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


def last_next_coupon(trade_dt: date, schedule: List[date]) -> Tuple[date, date]:
    last_c = schedule[0]
    for d in schedule:
        if d <= trade_dt:
            last_c = d
        else:
            return last_c, d
    return last_c, schedule[-1]


# --------------------------------------------------------------------------------------
# Bonds
# --------------------------------------------------------------------------------------
def _gen_bonds(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    today = date.today()

    # static fields
    issuer_ids = rng.integers(1, 400, size=n)
    sectors = rng.choice(SECTORS, size=n)
    ratings = rng.choice(RATINGS, size=n, p=[0.02, 0.10, 0.28, 0.35, 0.18, 0.07])
    coupons = rng.normal(3.2, 1.2, size=n).clip(0.5, 10.0)
    issue_age = rng.uniform(0.1, 7.0, size=n)
    issue_dates = [today - timedelta(days=int(365.25 * float(a))) for a in issue_age]
    maturities = [maturity_from_issue(d) for d in issue_dates]
    tenors = [max(0.25, (md - today).days / 365.25) for md in maturities]
    freq = rng.choice(FREQ_CHOICES, size=n)
    day_counts = rng.choice(DAY_COUNTS, size=n)

    # curves / pricing
    base_curve_yield = np.array([ns_yield(t) for t in tenors], dtype=float)
    # sector/rating levels for "clean" anchor
    sector_levels = {s: i for i, s in enumerate(sorted(set(sectors.tolist())))}
    rating_levels = {r: i for i, r in enumerate(sorted(RATINGS))}

    sector_code = np.array([sector_levels[s] for s in sectors], dtype=float)
    rating_code = np.array([rating_levels[r] for r in ratings], dtype=float)

    clean_delta_bps = (
        0.0
        + 3.0 * sector_code
        + 5.0 * rating_code
        + rng.normal(0.0, 2.0, size=n)
    )
    price0_clean = 100.0 * (1.0 - clean_delta_bps / 10_000.0)

    # OAS / yield / duration / dv01
    oas0_bps = (50 + 15 * rating_code + rng.normal(0.0, 12.0, size=n)).astype(float)
    ytm0 = base_curve_yield + oas0_bps / 10_000.0
    mdur = np.zeros(n, dtype=float)
    dv01 = np.zeros(n, dtype=float)
    p_chk = np.zeros(n, dtype=float)
    for i in range(n):
        p, m, d = mod_duration_price_dv01(coupons[i], ytm0[i], tenors[i], int(freq[i]))
        p_chk[i] = p
        mdur[i] = m
        dv01[i] = d

    # amount outstanding and coarse curve bucket (kept for backward-compatibility)
    amount_out = rng.lognormal(mean=8.2, sigma=0.7, size=n)  # ~ $3.6m median
    curve_bucket_coarse = rng.choice(CURVE_BUCKETS_COARSE, size=n)
    curve_bucket_fine = np.array([tenor_bucket_fine(t) for t in tenors], dtype=object)

    # vendor static liquidity (used to scale intensity and as baseline for providers)
    vendor_liq_static = np.clip(
        50
        + (np.log10(amount_out) - 6.0) * 12.0
        + (4 - rating_code) * 5.0
        + rng.normal(0, 6.0, size=n),
        1,
        99,
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
        "curve_bucket": curve_bucket_coarse.tolist(),   # (compat field)
        # --- new/enhanced static fields ---
        "curve_bucket_fine": curve_bucket_fine.tolist(),
        "coupon_frequency": [int(f) for f in freq],
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
    For each provider p in params.providers generate a vendor liquidity score and a
    monotone gp mapping to a baseline premium π_ref in bps.
    Returns: (bonds_with_provider_cols, active_provider_name)
    """
    rng = np.random.default_rng(params.seed + 77)
    bonds = bonds.copy()

    # choose active provider (used to form the observed price)
    active_provider = params.providers[0] if len(params.providers) else "P1"

    for p in params.providers:
        # provider-specific noise/scale on top of the static
        liq = np.clip(
            bonds["vendor_liq_score_static"].to_numpy()
            + rng.normal(0.0, 4.0, size=len(bonds)),
            1, 99
        )
        bonds[f"vendor_{p}_liq"] = liq
        # monotone linear map (intercept + slope * (L/100))
        pi_ref = params.gp_intercept_bps + params.gp_slope_bps * (liq / 100.0)
        bonds[f"pi_ref_{p}_bps"] = pi_ref.astype(float)

    bonds["active_provider"] = active_provider
    return bonds, active_provider


# --------------------------------------------------------------------------------------
# Trade generation utilities
# --------------------------------------------------------------------------------------
def _pick_trade_count_per_bond(bonds: pd.DataFrame, params: SimParams, rng: np.random.Generator) -> Dict[str, int]:
    """Poisson count with intensity scaled by vendor static liquidity."""
    lref = bonds["vendor_liq_score_static"].to_numpy()
    lam = params.base_intensity + params.liq_to_intensity * (lref / 100.0)
    lam = np.clip(lam, 0.01, None)
    counts = rng.poisson(lam, size=len(bonds))
    return {bonds.loc[i, "isin"]: int(counts[i]) for i in range(len(bonds))}


def _cap_threshold(rating: str, params: SimParams) -> float:
    return params.cap_ig if rating in ("AAA", "AA", "A", "BBB") else params.cap_hy


def _settlement_date(trade_date: date, days: int = 2) -> date:
    # Simplified T+2 (no holiday calendar)
    return trade_date + timedelta(days=days)


def _portfolio_patterns() -> List[str]:
    return ["sector_crowding", "issuer_crowding", "curve_imbalance", "balanced"]


def _portfolio_delta(
    pattern: str,
    side_sign: float,
    port_skew: float,
    bond_row: pd.Series,
    portfolio_focus: Dict[str, object],
    rng: np.random.Generator,
) -> float:
    """
    Compute a portfolio-conditioned delta (bps) for a single line, conditioned on the basket's
    pattern and the line's characteristics.
    Positive means wider (worse) for SELLs; negative (tighter) for BUYs.
    """
    # base strength bounded to positive
    strength = max(0.05, abs(port_skew))

    if pattern == "sector_crowding":
        # lines in the focus sector get larger adjustment, others smaller
        same_sector = float(bond_row["sector"] == portfolio_focus["sector"])
        delta = (1.25 * same_sector + 0.6 * (1 - same_sector)) * strength

    elif pattern == "issuer_crowding":
        # heavier adjustment for the top issuer of the basket
        same_issuer = float(bond_row["issuer"] == portfolio_focus["issuer"])
        delta = (1.5 * same_issuer + 0.5 * (1 - same_issuer)) * strength

    elif pattern == "curve_imbalance":
        # emphasize whichever curve bucket is overrepresented
        same_curve = float(bond_row["curve_bucket_fine"] == portfolio_focus["curve_bucket_fine"])
        delta = (1.35 * same_curve + 0.7 * (1 - same_curve)) * strength

    else:  # "balanced" or unknown
        delta = 0.9 * strength

    # Direction: SELL widens, BUY tightens
    return float(side_sign * delta * 1.0)  # in bps


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
    Build the truthful realized premium y_bps = π_ref + h(Q,U) + Δ + ε.
    Return (y_bps, h_bps, eps_bps, delta_bps, pi_ref_bps)
    """
    h_bps = (
        params.liq_size_coeff * size_z
        + params.liq_side_coeff * side_sign
        + params.liq_sector_coeff * sec_code
        + params.liq_rating_coeff * rat_code
        + params.liq_urgency_coeff * urgency
    )
    eps_bps = rng.normal(0.0, params.liq_eps_bps)
    y_bps = float(baseline_ref_bps + h_bps + port_delta_bps + eps_bps)
    return y_bps, float(h_bps), float(eps_bps), float(port_delta_bps), float(baseline_ref_bps)


def _sale_condition3(is_late: bool, after_hours: bool, rng: np.random.Generator) -> str:
    if is_late and after_hours:
        return "U"  # Late After Hours
    if is_late:
        return "Z"  # Late
    if after_hours:
        return "T"  # After Hours
    return ""


# --------------------------------------------------------------------------------------
# Trades (+ truthful y_bps) using bonds' clean_price reference
# --------------------------------------------------------------------------------------
def _gen_trades_with_targets(bonds: pd.DataFrame, n_days: int, params: SimParams) -> pd.DataFrame:
    rng = np.random.default_rng(params.seed + 101)
    start = datetime(2025, 1, 2, 9, 0, 0)

    # Provider baselines
    bonds2, active_provider = _providers_liq_and_pref(params, bonds)
    provider_act_col = f"pi_ref_{active_provider}_bps"
    provider_liq_col = f"vendor_{active_provider}_liq"

    # category codes for generator
    sector_levels = {s: i for i, s in enumerate(sorted(bonds2["sector"].unique().tolist()))}
    rating_levels = {r: i for i, r in enumerate(sorted(RATINGS))}

    trades_rows = []

    for d in range(n_days):
        trade_date = (start + timedelta(days=int(d))).date()

        # Draw how many times each bond trades today
        counts = _pick_trade_count_per_bond(bonds2, params, rng)

        # Expand to a list of ISINs to execute in random order
        todays_isins = [isin for isin, k in counts.items() for _ in range(k)]
        if not todays_isins:
            continue
        rng.shuffle(todays_isins)

        # Pre-allocate a set of indices that will be assigned to portfolio baskets
        n_total = len(todays_isins)
        approx_port_lines = int(params.portfolio_trade_share * n_total)
        used = set()

        # Determine number of baskets and their sizes
        baskets: List[List[int]] = []
        remaining = approx_port_lines
        while remaining >= params.basket_size_min:
            bsize = int(rng.integers(params.basket_size_min, params.basket_size_max + 1))
            if bsize > remaining:
                bsize = remaining
            baskets.append([])
            remaining -= bsize

        # Fill baskets by sampling indices with preference for same sector proximity
        # We select seeds and then add similar bonds (same sector, close rating).
        for b in baskets:
            # pick a seed index not yet used
            candidates = [i for i in range(n_total) if i not in used]
            if not candidates:
                break
            i0 = int(rng.choice(candidates))
            used.add(i0)
            b.append(i0)

            seed_isin = todays_isins[i0]
            seed_b = bonds2.loc[bonds2["isin"] == seed_isin].iloc[0]

            # gather pool of unused indices with same sector preference
            pool = [i for i in candidates if i != i0]
            rng.shuffle(pool)

            def score(i: int) -> float:
                bb = bonds2.loc[bonds2["isin"] == todays_isins[i]].iloc[0]
                s = 0.0
                s += 1.0 if bb["sector"] == seed_b["sector"] else 0.0
                s += 0.5 * (1.0 - abs(float(rating_levels[bb["rating"]]) - float(rating_levels[seed_b["rating"]])) / 5.0)
                s += 0.2 * (1.0 if bb["curve_bucket_fine"] == seed_b["curve_bucket_fine"] else 0.0)
                return s

            # pick required basket size - 1 more (seed already counted)
            need = max(bsize - 1, 0)
            # try to fill to need (or as much as we can)
            pool_sorted = sorted(pool, key=score, reverse=True)[:max(need, 1) * 4]
            for i in pool_sorted:
                if len(b) >= need + 1:  # seed + need
                    break
                if i not in used:
                    used.add(i)
                    b.append(i)

        # Build an index→basket_id map
        index_to_port: Dict[int, int] = {}
        for b_id, b in enumerate(baskets):
            for idx in b:
                index_to_port[idx] = b_id

        # Now generate trades in sequence
        seq_counter = 1
        for i, isin in enumerate(todays_isins):
            bnd = bonds2.loc[bonds2["isin"] == isin].iloc[0]
            base_time = datetime.combine(trade_date, datetime.min.time()) + timedelta(minutes=int(rng.integers(9*60, 16*60)))

            in_portfolio = i in index_to_port
            basket_id = index_to_port.get(i, None)
            side = rng.choice(["BUY", "SELL"])
            side_sign = 1.0 if side == "SELL" else -1.0

            # Sizes & capping
            if bnd["rating"] in ("AAA","AA","A","BBB"):
                mu = math.log(1.2e6); sigma = 1.0
            else:
                mu = math.log(6.0e5); sigma = 1.0
            size = float(np.clip(rng.lognormal(mu, sigma), 25_000.0, 10_000_000.0))
            # round to 5k
            size = float(int(size/5000)*5000)
            cap_thr = _cap_threshold(str(bnd["rating"]), params)
            is_capped = size > cap_thr
            reported_size = min(size, cap_thr) if is_capped else size

            # urgency ~ U[0,1)
            urgency = float(rng.random())

            # vendor baseline for active provider
            pi_ref_bps = float(bnd[provider_act_col])

            # h(Q,U) components need standardization by ISIN (size_z)
            # we'll compute running mean/std by ISIN within this day scope in a simple way:
            # for a stable approximation, treat the per-ISIN distribution as lognormal; compute z via a rolling proxy.
            # Here we approximate by z = log(size) minus cross-sectional mean, for stationarity.
            # z-score relative to the lognormal draw parameters used above
            size_z = (math.log(max(1.0, size)) - mu) / (sigma + 1e-8)
            # category codes
            sec_code = float(sector_levels[str(bnd["sector"])])
            rat_code = float(rating_levels[str(bnd["rating"])])

            # Portfolio specifics
            portfolio_id = None
            sale_condition4 = ""
            port_delta_bps = 0.0
            port_skew = 0.0
            port_similarity = 0.0
            pattern = ""
            if in_portfolio and basket_id is not None:
                sale_condition4 = "P"
                portfolio_id = f"PF_{trade_date.strftime('%Y%m%d')}_{basket_id:04d}"
                # For the basket, determine its shared pattern & parameters deterministically
                rng_b = np.random.default_rng(params.seed + 10_000 + basket_id)
                pattern = rng_b.choice(_portfolio_patterns())
                port_skew = float(rng_b.normal(params.port_skew_mu, params.port_skew_sigma))
                focus = {
                    "sector": rng_b.choice(SECTORS),
                    "issuer": bnd["issuer"],  # lightweight proxy
                    "curve_bucket_fine": rng_b.choice(CURVE_BUCKETS_FINE),
                }
                # second-level time clustering around the first line's base_time
                base_time = base_time.replace(second=0, microsecond=0) if params.second_resolution else base_time
                jitter = int(rng_b.integers(-params.port_time_spread_sec, params.port_time_spread_sec + 1))
                base_time = base_time + timedelta(seconds=jitter)

                # compute portfolio-conditioned delta for this line
                port_delta_bps = _portfolio_delta(pattern, side_sign, port_skew, bnd, focus, rng_b)
                port_similarity = 1.0  # mark as fully portfolio-conditioned

            # Compose the truthful premium and observed prices
            y_bps, h_bps, eps_bps, delta_bps, pi_ref_used = _compose_y_bps(
                baseline_ref_bps=pi_ref_bps,
                size_z=size_z,
                side_sign=side_sign,
                sec_code=sec_code,
                rat_code=rat_code,
                urgency=urgency,
                params=params,
                port_delta_bps=port_delta_bps,
                rng=rng,
            )
            micro = float(rng.normal(0.0, params.micro_price_noise_bps))
            delta_obs_bps = y_bps + micro
            clean_price = float(bnd["price0_clean"])
            price_clean_exec = clean_price * (1.0 - delta_obs_bps / 10_000.0)

            # Accrued & dirty price (30/360 approximation back-of-envelope)
            freq = int(bnd["coupon_frequency"])
            sched = coupon_schedule(bnd["issue_date"].date(), bnd["maturity"].date(), freq)
            last_c, next_c = last_next_coupon(trade_date, sched)
            d1 = last_c; d2 = trade_date
            # 30/360 daycount simple
            D = (360*(d2.year-d1.year) + 30*(d2.month-d1.month) + (d2.day-d1.day)) / 360.0
            accrued = 100.0 * (float(bnd["coupon"])/100.0) * (D / (12.0/freq))
            price_dirty_exec = price_clean_exec + accrued

            # Yield & OAS proxies (invert approximately via DV01)
            dv01_per_100 = float(bnd["dv01_per_100"])
            dy = - (price_clean_exec - clean_price) / max(1e-8, dv01_per_100)  # per 1.00 notional
            ytm_exec = float(bnd["ytm0"] + dy)

            # flags and timestamps
            is_late = bool(rng.random() < params.late_rate)
            is_asof = bool(rng.random() < params.asof_rate)
            is_cancel = bool(rng.random() < params.cancel_rate)
            after_hours = bool((base_time.hour < 9) or (base_time.hour > 16))

            report_delay_min = int(rng.integers(0, 60 if is_late else 10))
            report_time = base_time + timedelta(minutes=report_delay_min)
            exec_time = base_time

            # TRACE-ish sale conditions
            sale_condition3 = _sale_condition3(is_late, after_hours, rng)
            remuneration = rng.choice(["", "C", "M", "N"], p=[0.7, 0.1, 0.1, 0.1])
            ats = bool(rng.random() < params.ats_rate)
            contra = "C"  # customer
            asof_indicator = "A" if is_asof else ""

            trades_rows.append(dict(
                # --- Back-compat core (unchanged names) ---
                ts=pd.to_datetime(report_time),   # historically used as the trade timestamp
                isin=str(bnd["isin"]),
                side=str(side),                   # "BUY"/"SELL"
                size=float(size),
                is_portfolio=bool(in_portfolio),

                # --- Truth decomposition ---
                y_bps=float(y_bps),               # realized premium (truth)
                price=float(price_clean_exec),    # observed CLEAN price
                y_h_bps=float(h_bps),
                y_eps_bps=float(eps_bps),
                y_delta_port_bps=float(delta_bps),
                y_pi_ref_bps=float(pi_ref_used),
                urgency=float(urgency),

                # --- TRACE-like & analytics ---
                exec_time=pd.to_datetime(exec_time),
                report_time=pd.to_datetime(report_time),
                trade_dt=pd.to_datetime(trade_date),
                is_late=bool(is_late),
                is_asof=bool(is_asof),
                is_cancel=bool(is_cancel),
                contra_party_type=str(contra),
                remuneration=str(remuneration),
                sale_condition3=str(sale_condition3),
                sale_condition4=str(sale_condition4),
                asof_indicator=str(asof_indicator),
                ats=bool(ats),

                reported_size=float(reported_size),
                trace_cap_indicator=bool(is_capped),
                cap_threshold=float(cap_thr),

                clean_price=float(clean_price),
                price_clean_exec=float(price_clean_exec),
                accrued_interest=float(accrued),
                price_dirty_exec=float(price_dirty_exec),
                yield_exec=float(ytm_exec),
                dv01_per_100=float(dv01_per_100),
                dv01_dollar=float(dv01_per_100 * (size/100.0)),
                dv01_signed=float(dv01_per_100 * (1 if side=="BUY" else -1)),

                sector=str(bnd["sector"]),        # convenience copy on trade row
                rating=str(bnd["rating"]),
                vendor_liq_score=float(bnd[provider_liq_col]),
                active_provider=str(active_provider),

                # portfolio labeling
                portfolio_id=str(portfolio_id) if portfolio_id else None,
                portfolio_pattern=str(pattern) if pattern else "",
                portfolio_skew=float(port_skew),
                portfolio_similarity=float(port_similarity),
            ))
            seq_counter += 1

    if not trades_rows:
        # Return an empty frame with the core required columns to keep downstream code robust
        core_cols = [
            "ts","isin","side","size","is_portfolio","y_bps","price"
        ]
        return pd.DataFrame(columns=core_cols)

    trades = pd.DataFrame(trades_rows).sort_values(["exec_time","isin"]).reset_index(drop=True)
    return trades


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------
def simulate(params: SimParams) -> Dict[str, pd.DataFrame]:
    set_seed(params.seed)
    logging.getLogger(__name__).info("Simulating %d bonds over %d days", params.n_bonds, params.n_days)

    bonds = _gen_bonds(params.n_bonds, params.seed)
    trades = _gen_trades_with_targets(bonds, params.n_days, params)

    # Basic integrity checks
    assert {"isin","issuer","sector","rating","issue_date","maturity","coupon","amount_out","curve_bucket"}.issubset(bonds.columns), \
        "Required bond columns are missing"
    assert {"ts","isin","side","size","is_portfolio","y_bps","price"}.issubset(trades.columns), \
        "Required trade columns are missing"

    return {"bonds": bonds, "trades": trades}