from pathlib import Path
import hashlib
import pandas as pd

from ptliq.cli.simulate import main as sim_main


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def test_repro(tmp_path: Path):
    # run simulate twice with the same seed; then re-run with different seed
    run1 = tmp_path / "run1"
    run2 = tmp_path / "run2"
    run3 = tmp_path / "run3"

    # Use CLI entry to also generate schema/config sidecars
    # Use small sim size for speed
    sim_main(config=Path("configs/base.yaml"), outdir=run1, seed=123, n_bonds=80, n_days=5, loglevel="WARNING")
    sim_main(config=Path("configs/base.yaml"), outdir=run2, seed=123, n_bonds=80, n_days=5, loglevel="WARNING")
    sim_main(config=Path("configs/base.yaml"), outdir=run3, seed=456, n_bonds=80, n_days=5, loglevel="WARNING")

    t1 = run1 / "trades.parquet"
    t2 = run2 / "trades.parquet"
    t3 = run3 / "trades.parquet"

    # Parquet must exist
    assert t1.exists() and t2.exists() and t3.exists()

    # Compare file hashes
    assert _sha_file(t1) == _sha_file(t2)
    assert _sha_file(t1) != _sha_file(t3)

    # Bonds too (sanity)
    b1 = run1 / "bonds.parquet"
    b2 = run2 / "bonds.parquet"
    b3 = run3 / "bonds.parquet"
    assert _sha_file(b1) == _sha_file(b2)
    assert _sha_file(b1) != _sha_file(b3)
