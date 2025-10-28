from __future__ import annotations

import pandas as pd
import pytest

from ptliq.data.simulate import SimParams, simulate


@pytest.fixture(scope="module")
def sim_df(tmp_path_factory: pytest.TempPathFactory) -> pd.DataFrame:
    # Small simulation; turn off legacy h(Q,U) to make residual independent of size/side by construction
    out = tmp_path_factory.mktemp("sim_out")
    params = SimParams(
        n_bonds=200,
        n_days=8,
        providers=["P1"],
        seed=1234,
        outdir=out,
        # Deterministic size/side in pi_ref via y_h_bps; residual excludes it
        liq_size_coeff=1.0,
        liq_side_coeff=1.0,
        liq_sector_coeff=0.0,
        liq_rating_coeff=0.0,
        liq_eps_bps=2.0,
        # Remove portfolio and Δ*(P) effects to keep residual independent
        portfolio_trade_share=0.0,
        delta_scale=0.0,
        delta_bias=0.0,
        delta_size=0.0,
        delta_side=0.0,
        delta_issuer=0.0,
        delta_sector=0.0,
        delta_noise_std=0.0,
    )
    frames = simulate(params)
    trades = frames["trades"].copy()
    # For this test, we expect a numeric 'side' column. Provide one by mapping.
    # Do not mutate original columns used elsewhere.
    if trades["side"].dtype == object:
        side_sign = trades.get("side_sign")
        if side_sign is None:
            side_sign = trades["side"].map({"SELL": 1.0, "CSELL": 1.0, "BUY": -1.0, "CBUY": -1.0})
        trades = trades.copy()
        trades["side"] = pd.to_numeric(side_sign, errors="coerce").astype(float)
    # Ensure quantity_par column exists for size-based correlation
    if "quantity_par" not in trades.columns:
        trades["quantity_par"] = pd.to_numeric(trades.get("size", 0.0), errors="coerce")
    # Use full set for better correlation stability
    return trades.reset_index(drop=True)


def test_sim_outputs_have_contract(sim_df: pd.DataFrame):  # sim_df: small dataframe created by simulate.py
    assert {"y_bps","pi_ref_bps","residual_bps"} <= set(sim_df.columns)
    # residual has near-zero correlation with side/size by design
    import numpy as np
    side = np.asarray(sim_df["side"], float)
    log_size = np.log1p(np.abs(np.asarray(sim_df["quantity_par"], float)))
    r = np.asarray(sim_df["residual_bps"], float)
    def corr(x,y): 
        m = np.isfinite(x)&np.isfinite(y)
        x,y=x[m],y[m]
        x=(x-x.mean())/x.std(); y=(y-y.mean())/y.std()
        return float(np.nan if (len(x)<3 or x.std()==0 or y.std()==0) else (x*y).mean())
    assert abs(corr(r, side)) < 0.1
    assert abs(corr(r, log_size)) < 0.1
