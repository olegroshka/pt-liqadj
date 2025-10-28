from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from ptliq.data.simulate import SimParams, simulate
from ptliq.training.gat_loop import SamplerConfig, build_samples


def _corr(x: np.ndarray | pd.Series, y: np.ndarray | pd.Series) -> float:
    if isinstance(x, pd.Series):
        x = pd.to_numeric(x, errors="coerce").to_numpy()
    if isinstance(y, pd.Series):
        y = pd.to_numeric(y, errors="coerce").to_numpy()
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10:
        return 0.0
    try:
        return float(pearsonr(x[m], y[m])[0])
    except Exception:
        return 0.0


def _make_nodes_from_bonds(bonds: pd.DataFrame) -> pd.DataFrame:
    nodes = bonds.copy()
    nodes = nodes.reset_index(drop=True)
    nodes["node_id"] = np.arange(len(nodes), dtype=int)
    # Ensure required identifiers are present
    for c in ["isin", "issuer", "sector", "curve_bucket", "currency"]:
        if c not in nodes.columns:
            nodes[c] = ""
    return nodes[["isin", "node_id", "issuer", "sector", "curve_bucket", "currency"]]


def test_build_samples_invariants_and_probes(tmp_path: Path):
    # --- Simulate a small dataset with Phase-1 Δ*(P) turned on and legacy h(Q,U) off ---
    params = SimParams(
        n_bonds=120,
        n_days=6,
        providers=["P1"],
        seed=202,
        outdir=tmp_path,
        # Zero legacy h(Q,U) so residual ~ Δ*(P) + ε
        liq_size_coeff=0.0,
        liq_side_coeff=0.0,
        liq_sector_coeff=0.0,
        liq_rating_coeff=0.0,
        liq_eps_bps=1.0,
        # Planted Δ*(P) coefficients
        delta_scale=1.0,
        delta_bias=0.0,
        delta_size=8.0,
        delta_side=10.0,
        delta_issuer=6.0,
        delta_sector=4.0,
        delta_noise_std=2.0,
    )
    frames = simulate(params)
    bonds, trades = frames["bonds"], frames["trades"]

    # Basic sanity: we should have some portfolio trades to form baskets
    assert trades["portfolio_id"].notna().sum() >= 10, "Simulation produced too few portfolio trades for sampling"

    # Build nodes and samples
    nodes = _make_nodes_from_bonds(bonds)
    sampler = SamplerConfig(seed=777, weight_mode="signed_frac", coalesce_pf_base=False)
    samples, diags = build_samples(trades, nodes, sampler)

    # --- Invariants ---
    assert len(samples) > 0, "No samples built — check grouping logic"

    # Residual distribution not degenerate
    res = np.array([s.residual for s in samples], dtype=float)
    assert np.nanstd(res) > 1e-6

    # base_feats present and length 2 = [log_size, side]
    for s in samples:
        assert isinstance(s.base_feats, (list, tuple)) and len(s.base_feats) == 2, "base_feats must be length-2"
        assert np.isfinite(s.base_feats[0]), "log_size must be finite"
        assert np.isfinite(s.base_feats[1]), "side sign must be finite"

    # port_nodes and port_weights lengths match; abs weight fractions sum to ~1
    for s in samples:
        assert len(s.port_nodes) == len(s.port_weights), "port_nodes and port_weights length mismatch"
        if len(s.port_weights) > 0:
            w_abs_sum = float(np.sum(np.abs(np.array(s.port_weights, dtype=float))))
            assert abs(w_abs_sum - 1.0) <= 1e-3, f"abs weight fractions must sum to 1, got {w_abs_sum}"

    # --- Probes and correlation checks ---
    # Phase-1 target signs from params
    theta_size = params.delta_size
    theta_side = params.delta_side
    theta_issuer = params.delta_issuer

    # Sample-level correlations from diagnostics returned by build_samples
    r_s = np.asarray(diags["residual"], dtype=float)
    c_log_size_s = _corr(np.asarray(diags["log_size"], dtype=float), r_s)
    c_side_s = _corr(np.asarray(diags["side"], dtype=float), r_s)
    c_frac_iss_s = _corr(np.asarray(diags["frac_same_issuer"], dtype=float), r_s)
    c_sum_abs_s = _corr(np.asarray(diags["sum_abs_w"], dtype=float), r_s)
    c_sum_signed_s = _corr(np.asarray(diags["sum_signed_w"], dtype=float), r_s)
    c_vendor_liq_s = _corr(np.asarray(diags["vendor_liq"], dtype=float), r_s)

    # Check signs align with planted coefficients (allow near-zero if unstable)
    assert (np.sign(c_log_size_s) == np.sign(theta_size)) or (abs(c_log_size_s) < 0.05), f"Unexpected corr sign for log_size: {c_log_size_s}"
    assert (np.sign(c_side_s) == np.sign(theta_side)) or (abs(c_side_s) < 0.05), f"Unexpected corr sign for side: {c_side_s}"
    if theta_issuer > 0:
        assert (c_frac_iss_s > 0) or (abs(c_frac_iss_s) < 0.05), f"frac_same_issuer corr should be >=0, got {c_frac_iss_s}"
    else:
        assert (c_frac_iss_s < 0) or (abs(c_frac_iss_s) < 0.05), f"frac_same_issuer corr should be <=0, got {c_frac_iss_s}"

    # Compare to correlations computed directly on simulated trades for consistency (±0.05)
    # Use portfolio trades subset to mirror sampling groups
    pf = trades[trades["portfolio_id"].notna()].copy()
    r_t = pd.to_numeric(pf["residual"], errors="coerce")
    c_log_size_t = _corr(pd.to_numeric(pf.get("log_size"), errors="coerce"), r_t)
    # prefer side_sign if present
    side_series = pf.get("side_sign", pf.get("sign"))
    c_side_t = _corr(pd.to_numeric(side_series, errors="coerce"), r_t)
    c_frac_iss_t = _corr(pd.to_numeric(pf.get("frac_same_issuer"), errors="coerce"), r_t)

    assert abs(c_log_size_s - c_log_size_t) <= 0.05, f"Sample vs trade corr(log_size) differ: {c_log_size_s} vs {c_log_size_t}"
    assert abs(c_side_s - c_side_t) <= 0.05, f"Sample vs trade corr(side) differ: {c_side_s} vs {c_side_t}"
    # frac_same_issuer is noisier; still aim for ±0.05
    assert abs(c_frac_iss_s - c_frac_iss_t) <= 0.05, f"Sample vs trade corr(frac_same_issuer) differ: {c_frac_iss_s} vs {c_frac_iss_t}"

    # Sensible ranges for other probes
    # Vendor liq is excluded from residual by construction -> small correlation
    assert abs(c_vendor_liq_s) <= 0.2
    # Sum of absolute context not directly in Δ*(P) -> correlation should not dominate
    assert abs(c_sum_abs_s) <= 0.5
    # Signed sum often near-neutral by design -> modest magnitude
    assert abs(c_sum_signed_s) <= 0.3
