import pandas as pd
import numpy as np
from pathlib import Path

from ptliq.cli.featurize import _compute_portfolio_weights


def make_trades():
    # Two baskets on same day, each >=2 lines
    dt = pd.to_datetime("2024-07-15")
    data = [
        {"portfolio_id": "PF_20240715_A1", "trade_dt": dt, "isin": "BOND1", "dv01_dollar": 100.0, "side": "BUY"},
        {"portfolio_id": "PF_20240715_A1", "trade_dt": dt, "isin": "BOND2", "dv01_dollar": 50.0,  "side": "BUY"},
        {"portfolio_id": "PF_20240715_A1", "trade_dt": dt, "isin": "BOND3", "dv01_dollar": 50.0,  "side": "BUY"},
        {"portfolio_id": "PF_20240715_B9", "trade_dt": dt, "isin": "BOND4", "dv01_dollar": 40.0,  "side": "SELL"},
        {"portfolio_id": "PF_20240715_B9", "trade_dt": dt, "isin": "BOND5", "dv01_dollar": 60.0,  "side": "SELL"},
    ]
    return pd.DataFrame(data)


def test_portfolio_weights_unit_sum_and_bounds():
    trades = make_trades()
    node_id_map = {f"BOND{i}": i-1 for i in range(1, 6)}

    pl = _compute_portfolio_weights(trades, node_id_map)
    assert not pl.empty
    # per-group unit sum
    sums = pl.groupby("pf_gid")["w_dv01_abs_frac"].sum().to_numpy()
    assert np.allclose(sums, 1.0, atol=1e-8)
    # bounds and no NaNs
    assert not pl["w_dv01_abs_frac"].isna().any()
    assert (pl["w_dv01_abs_frac"] >= 0).all()
    assert (pl["w_dv01_abs_frac"] <= 1).all()

    # signed fractions reflect side
    # First basket BUY -> positive
    g0 = pl[pl["pf_gid"] == pl["pf_gid"].min()]
    assert (g0["w_dv01_signed_frac"] >= 0).all()
    # Second basket SELL -> negative
    g1 = pl[pl["pf_gid"] == pl["pf_gid"].max()]
    assert (g1["w_dv01_signed_frac"] <= 0).all()


def test_pf_gid_deterministic_ordering():
    trades = make_trades()
    node_id_map = {f"BOND{i}": i-1 for i in range(1, 6)}

    # Shuffle rows; pf_gid assignment should be stable per (portfolio_id, trade_dt)
    pl1 = _compute_portfolio_weights(trades.sample(frac=1.0, random_state=7).reset_index(drop=True), node_id_map)
    pl2 = _compute_portfolio_weights(trades.sample(frac=1.0, random_state=13).reset_index(drop=True), node_id_map)

    # Map pf_gid to (portfolio_id,trade_dt)
    idx1 = pl1[["pf_gid", "portfolio_id", "trade_dt"]].drop_duplicates().sort_values("pf_gid").reset_index(drop=True)
    idx2 = pl2[["pf_gid", "portfolio_id", "trade_dt"]].drop_duplicates().sort_values("pf_gid").reset_index(drop=True)

    pd.testing.assert_frame_equal(idx1, idx2)
