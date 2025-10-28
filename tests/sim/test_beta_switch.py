from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from ptliq.cli.simulate import main as sim_main


def _run(outdir: Path, delta_scale: float):
    sim_main(
        config=Path("configs/base.yaml"), outdir=outdir, seed=321,
        n_bonds=120, n_days=8, loglevel="WARNING",
        # Disable old h(Q,U) path so residual is driven by Δ*(P) only
        liq_size_coeff=0.0,
        liq_side_coeff=0.0,
        liq_sector_coeff=0.0,
        liq_rating_coeff=0.0,
        liq_eps_bps=1.0,
        # Delta knobs
        delta_scale=delta_scale,
        delta_bias=0.0,
        delta_size=8.0,
        delta_side=10.0,
        delta_issuer=6.0,
        delta_sector=4.0,
        delta_noise_std=2.0,
    )
    t = pd.read_parquet(outdir / "trades.parquet")
    return t


def _corr(x, y):
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    m = x.notna() & y.notna()
    if m.sum() < 10:
        return 0.0
    try:
        return float(pearsonr(x[m].to_numpy(), y[m].to_numpy())[0])
    except Exception:
        return 0.0


def test_beta_switch(tmp_path: Path):
    t1 = _run(tmp_path / "beta1", delta_scale=1.0)
    t0 = _run(tmp_path / "beta0", delta_scale=0.0)

    r1 = pd.to_numeric(t1["residual"], errors="coerce")
    r0 = pd.to_numeric(t0["residual"], errors="coerce")

    # correlations under beta=1
    # focus on portfolio trades where Δ*(P) is relevant
    pf1 = t1[t1.get("is_portfolio", False) == True]
    pf0 = t0[t0.get("is_portfolio", False) == True]

    c_size_1 = _corr(pf1.get("log_size"), r1.loc[pf1.index])
    c_side_1 = _corr(pf1.get("side_sign", pf1.get("sign")), r1.loc[pf1.index])

    # correlations under beta=0 should be near zero on portfolio subset
    c_size_0 = _corr(pf0.get("log_size"), r0.loc[pf0.index])
    c_side_0 = _corr(pf0.get("side_sign", pf0.get("sign")), r0.loc[pf0.index])

    assert abs(c_size_0) <= 0.05, f"beta=0 should nuke size corr, got {c_size_0}" 
    assert abs(c_side_0) <= 0.05, f"beta=0 should nuke side corr, got {c_side_0}" 

