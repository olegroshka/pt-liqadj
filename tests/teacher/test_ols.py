from pathlib import Path
import numpy as np
import pandas as pd

from ptliq.cli.simulate import main as sim_main


def _run_sim(outdir: Path):
    sim_main(
        config=Path("configs/base.yaml"), outdir=outdir, seed=777,
        n_bonds=120, n_days=8, loglevel="WARNING",
        # isolate Δ*(P)
        liq_size_coeff=0.0,
        liq_side_coeff=0.0,
        liq_sector_coeff=0.0,
        liq_rating_coeff=0.0,
        liq_eps_bps=1.0,
        # delta settings
        delta_scale=1.0,
        delta_bias=0.0,
        delta_size=8.0,
        delta_side=10.0,
        delta_issuer=6.0,
        delta_sector=4.0,
        delta_noise_std=2.0,
    )
    return pd.read_parquet(outdir / "trades.parquet")


def _ols_fit_predict(X_train, y_train, X_test):
    # add intercept
    Xtr = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    Xte = np.c_[np.ones((X_test.shape[0], 1)), X_test]
    # closed-form OLS with ridge eps for stability
    reg = 1e-6
    XtX = Xtr.T @ Xtr + reg * np.eye(Xtr.shape[1])
    Xty = Xtr.T @ y_train
    beta = np.linalg.solve(XtX, Xty)
    y_pred = Xte @ beta
    return y_pred


def test_ols_teacher(tmp_path: Path):
    t = _run_sim(tmp_path / "ols")
    # Features
    def _col(df, name, default=0.0):
        s = df[name] if name in df.columns else pd.Series(default, index=df.index)
        return pd.to_numeric(s, errors="coerce").fillna(default)

    feats = [
        _col(t, "log_size"),
        _col(t, "side_sign" if "side_sign" in t.columns else "sign"),
        _col(t, "frac_same_issuer"),
        _col(t, "sector_signed_conc"),
    ]
    X = np.column_stack([f.to_numpy() for f in feats])
    y = pd.to_numeric(t["residual"], errors="coerce").fillna(0.0).to_numpy()

    # Time split 80/20 by trade_dt
    dts = pd.to_datetime(t["trade_dt"]).dt.normalize()
    uniq = sorted(dts.dropna().unique())
    split_idx = int(0.8 * len(uniq))
    dt_train = set(uniq[:split_idx])

    mask_train = dts.isin(dt_train).to_numpy()
    mask_test = ~mask_train

    X_train, y_train = X[mask_train], y[mask_train]
    X_test, y_test = X[mask_test], y[mask_test]

    # Guard: ensure we have samples
    assert X_train.shape[0] > 50 and X_test.shape[0] > 20

    y_pred = _ols_fit_predict(X_train, y_train, X_test)

    # Metrics
    resid = y_test - y_pred
    mae = float(np.mean(np.abs(resid)))
    # R^2
    ss_res = float(np.sum((y_test - y_pred) ** 2))
    ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot

    assert r2 >= 0.4, f"OLS teacher too weak: R^2={r2:.3f}"
    assert mae < 8.0, f"OLS teacher MAE too high: {mae:.2f} bps"
