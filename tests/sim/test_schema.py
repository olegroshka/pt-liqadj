from pathlib import Path
import json
import pandas as pd

from ptliq.cli.simulate import main as sim_main


REQUIRED_TRADES = [
    "isin", "dv01_dollar", "exec_time", "trade_dt", "vendor_liq",
    "y_bps", "y_pi_ref_bps", "residual", "side"
]
# We accept either string side with companion side_sign, or numeric side

REQUIRED_BONDS = [
    "isin", "issuer", "sector", "rating", "curve_bucket", "maturity", "amount_out", "coupon"
]


def test_schema_and_sidecars(tmp_path: Path):
    outdir = tmp_path / "schema_run"
    sim_main(config=Path("configs/base.yaml"), outdir=outdir, seed=123, n_bonds=80, n_days=5, loglevel="WARNING")

    # Sidecars
    schema_path = outdir / "schema.json"
    cfg_path = outdir / "sim_config_used.json"
    assert schema_path.exists(), "schema.json must be written"
    assert cfg_path.exists(), "sim_config_used.json must be written"

    # Load parquet
    trades = pd.read_parquet(outdir / "trades.parquet")
    bonds = pd.read_parquet(outdir / "bonds.parquet")

    # Required bonds columns
    for c in REQUIRED_BONDS:
        assert c in bonds.columns, f"Missing bonds column: {c}"

    # Required trades columns (with side handling)
    for c in [x for x in REQUIRED_TRADES if x != "side"]:
        assert c in trades.columns, f"Missing trades column: {c}"
    # side must either be BUY/SELL string with side_sign present, or numeric +/-1
    assert "side" in trades.columns
    if trades["side"].dtype == object:
        assert "side_sign" in trades.columns
        # values should be in {-1, 1}
        ss = pd.to_numeric(trades["side_sign"], errors="coerce").dropna()
        assert set(ss.unique()).issubset({-1.0, 1.0})
    else:
        # numeric side column
        sv = pd.to_numeric(trades["side"], errors="coerce").dropna()
        assert set(sv.unique()).issubset({-1.0, 1.0})
