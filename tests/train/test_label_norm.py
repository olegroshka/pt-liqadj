from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from ptliq.data.simulate import SimParams, simulate
from ptliq.training.gat_loop import compute_label_norm


def _split_by_time(trades: pd.DataFrame, frac: float = 0.8):
    dts = pd.to_datetime(trades["trade_dt"]).dt.normalize()
    uniq = sorted(dts.dropna().unique())
    k = max(1, int(frac * len(uniq)))
    train_days = set(uniq[:k])
    m_tr = dts.isin(train_days)
    m_va = ~m_tr
    return m_tr.to_numpy(), m_va.to_numpy()


def test_label_norm_and_denorm_equivalence(tmp_path: Path):
    # Small, fast sim; exact sigma isn't prescribed — we validate equivalence and consistency
    params = SimParams(
        n_bonds=80,
        n_days=6,
        providers=["P1"],
        seed=11,
        outdir=tmp_path,
        # simplify legacy path to focus on residual composition
        liq_size_coeff=0.0,
        liq_side_coeff=0.0,
        liq_sector_coeff=0.0,
        liq_rating_coeff=0.0,
        liq_eps_bps=1.0,
        # planted Δ controls (kept modest so residual std is in a reasonable band)
        delta_scale=1.0,
        delta_bias=0.0,
        delta_size=6.0,
        delta_side=8.0,
        delta_issuer=5.0,
        delta_sector=0.0,
        delta_noise_std=2.0,
    )
    frames = simulate(params)
    t = frames["trades"]

    # Train/val split by day
    m_tr, m_va = _split_by_time(t, frac=0.8)
    r_tr = pd.to_numeric(t.loc[m_tr, "residual"], errors="coerce").fillna(0.0).to_numpy()
    r_va = pd.to_numeric(t.loc[m_va, "residual"], errors="coerce").fillna(0.0).to_numpy()

    # Label normalization per training loop helper
    mu, sigma = compute_label_norm(r_tr)

    # Recompute using the same formula inline to ensure identity (within floating noise)
    med = float(np.nanmedian(r_tr)) if r_tr.size > 0 else 0.0
    mad = float(np.nanmedian(np.abs(r_tr - med))) * 1.4826 if r_tr.size > 0 else 0.0
    sigma2 = float(mad if mad > 1e-6 else (np.nanstd(r_tr) + 1e-6))
    assert abs(sigma - sigma2) < 1e-9

    # Equivalence: MAE in bps equals MAE in normalized units times sigma
    # Use a simple constant predictor equal to the train mean (mu)
    y_true = r_va
    y_hat_bps = np.full_like(y_true, fill_value=mu)
    mae_bps = float(np.mean(np.abs(y_true - y_hat_bps))) if y_true.size > 0 else 0.0

    # Normalized-space equivalent predictor is zeros
    y_true_norm = (y_true - mu) / sigma if sigma > 0 else y_true * 0.0
    y_hat_norm = np.zeros_like(y_true_norm)
    mae_norm = float(np.mean(np.abs(y_true_norm - y_hat_norm))) if y_true_norm.size > 0 else 0.0

    assert abs(mae_bps - (mae_norm * sigma)) <= 1e-6, f"MAE bps {mae_bps:.6f} vs mae_norm*sigma {(mae_norm*sigma):.6f}"

    # Reasonable sigma range (sanity): avoid degenerate scaling
    assert 5.0 <= sigma <= 40.0
