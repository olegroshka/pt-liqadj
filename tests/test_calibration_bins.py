import numpy as np
from ptliq.backtest.metrics import calibration_bins

def test_calibration_bins_shape_and_monotonicity():
    yhat = np.linspace(-1.0, 1.0, 100)
    y = yhat + np.random.default_rng(0).normal(0, 0.1, size=100)
    df = calibration_bins(y, yhat, n_bins=10)
    assert len(df) == 10
    # bins are ordered 0..9
    assert df["bin"].min() == 0 and df["bin"].max() == 9
