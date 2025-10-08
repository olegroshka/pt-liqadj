import numpy as np
import pandas as pd
from ptliq.training.dataset import compute_standardizer, apply_standardizer

def test_standardizer_handles_zero_variance():
    df = pd.DataFrame({
        "f_const": [5.0]*10,
        "f_var": np.linspace(0, 9, 10),
        "y_bps": np.zeros(10),
    })
    feat_cols = ["f_const", "f_var"]
    stdz = compute_standardizer(df, feat_cols)
    assert np.all(stdz["std"] > 0)  # replaced zeros with 1.0
    X = apply_standardizer(df, feat_cols, stdz)
    # const column standardized to all zeros
    assert np.allclose(X[:, 0], 0.0)
