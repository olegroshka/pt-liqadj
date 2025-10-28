from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from ptliq.cli.simulate import main as sim_main


def _run_small_sim(outdir: Path, seed=123,
                   n_bonds=80, n_days=5,
                   delta_cfg=None,
                   zero_out_liq=True):
    kwargs = dict(
        config=Path("configs/base.yaml"), outdir=outdir, seed=seed,
        n_bonds=n_bonds, n_days=n_days, loglevel="WARNING",
    )
    # Zero out the old h(Q,U) path to isolate Δ*(P) correlations
    if zero_out_liq:
        kwargs.update(dict(
            liq_size_coeff=0.0,
            liq_side_coeff=0.0,
            liq_sector_coeff=0.0,
            liq_rating_coeff=0.0,
            liq_eps_bps=1.0,
        ))
    if delta_cfg:
        kwargs.update(delta_cfg)
    sim_main(**kwargs)
    trades = pd.read_parquet(outdir / "trades.parquet")
    return trades


def _corr_safely(x: pd.Series, y: pd.Series):
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 10:
        return np.nan, np.nan
    xr, yr = x[mask].to_numpy(), y[mask].to_numpy()
    pr = pearsonr(xr, yr)[0]
    sr = spearmanr(xr, yr)[0]
    return pr, sr


def test_residual_stats(tmp_path: Path):
    outdir = tmp_path / "phase1_stats"
    delta_cfg = dict(
        delta_scale=1.0,
        delta_bias=0.0,
        delta_size=8.0,
        delta_side=10.0,
        delta_issuer=6.0,
        delta_sector=4.0,
        delta_noise_std=2.0,
    )
    t = _run_small_sim(outdir, delta_cfg=delta_cfg)
    # Residual label
    r = pd.to_numeric(t["residual"], errors="coerce")

    # Basic spread (std) check: should be within a reasonable band
    std_bps = float(np.nanstd(r))
    assert 8.0 <= std_bps <= 30.0, f"Unexpected residual std: {std_bps}"

    # Correlation signs with planted features
    # log_size
    pr, sr = _corr_safely(t.get("log_size"), r)
    assert np.sign(pr) == np.sign(delta_cfg["delta_size"]) or abs(pr) < 0.02
    # side_sign
    pr_s, sr_s = _corr_safely(t.get("side_sign", t.get("sign")), r)
    assert np.sign(pr_s) == np.sign(delta_cfg["delta_side"]) or abs(pr_s) < 0.02
    # frac_same_issuer only for portfolio trades
    if "frac_same_issuer" in t.columns:
        pr_i, sr_i = _corr_safely(t["frac_same_issuer"], r)
        if delta_cfg["delta_issuer"] > 0:
            assert (pr_i > 0) or (abs(pr_i) < 0.05)
        else:
            assert (pr_i < 0) or (abs(pr_i) < 0.05)
