from pathlib import Path
import pandas as pd
from ptliq.data.simulate import SimParams, simulate

def test_simulate_shapes(tmp_path: Path):
    params = SimParams(n_bonds=50, n_days=3, providers=["P1"], seed=1, outdir=tmp_path)
    frames = simulate(params)
    assert {"bonds","trades"} <= set(frames.keys())
    b, t = frames["bonds"], frames["trades"]
    assert len(b) == 50
    assert {"isin","issuer","sector","rating","coupon"}.issubset(b.columns)
    assert {"ts","isin","side","size","price"}.issubset(t.columns)
    # write parquet to ensure IO layer works
    out_b = tmp_path / "bonds.parquet"
    b.to_parquet(out_b, index=False)
    df = pd.read_parquet(out_b)
    assert len(df) == 50
