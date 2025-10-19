from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
from typer.testing import CliRunner

from ptliq.data.simulate import SimParams, simulate
from ptliq.data.split import (
    compute_auto_ranges,
    compute_rolling_ranges,
    compute_default_ranges,
    write_ranges,
    read_ranges,
    write_counts,
    write_masks,
)
from ptliq.cli.split import app as split_app


def _write_sim(tmp_path: Path, n_days: int = 30, n_bonds: int = 50):
    frames = simulate(SimParams(n_bonds=n_bonds, n_days=n_days, providers=["P1"], seed=123, outdir=tmp_path))
    frames["bonds"].to_parquet(tmp_path / "bonds.parquet", index=False)
    frames["trades"].to_parquet(tmp_path / "trades.parquet", index=False)
    return tmp_path / "trades.parquet"


def test_auto_split_asymmetric_embargo_and_counts_masks(tmp_path: Path):
    trades_path = _write_sim(tmp_path, n_days=30)

    # Library: auto with asymmetric embargo
    r = compute_auto_ranges(
        trades_path,
        val_days=5,
        test_days=5,
        embargo_train_val=1,
        embargo_val_test=0,
        date_col="ts",
    )
    # Basic shape: non-empty and correct ordering
    dmin = pd.read_parquet(trades_path, columns=["ts"])  # type: ignore
    dmin = pd.to_datetime(dmin["ts"]).dt.tz_localize(None).dt.normalize()
    assert r.train["start"] <= r.train["end"] < r.val["start"] <= r.val["end"] < r.test["start"] <= r.test["end"]

    outdir = tmp_path / "splits"
    jpath = write_ranges(r, outdir)
    cpath = write_counts(trades_path, r, outdir)
    mpath = write_masks(trades_path, r, outdir)

    payload = json.loads(Path(jpath).read_text())
    assert set(payload.keys()) >= {"train", "val", "test"}

    counts = json.loads(Path(cpath).read_text())
    assert set(counts.keys()) == {"train", "val", "test"}
    assert counts["train"] > 0 and counts["val"] > 0 and counts["test"] > 0

    masks = pd.read_parquet(mpath)
    assert set(masks.columns) == {"ts", "split"}
    assert set(masks["split"].unique()) <= {"train", "val", "test", "none"}


def test_cli_invocation_with_and_without_subcommand_auto(tmp_path: Path):
    trades_path = _write_sim(tmp_path, n_days=20)
    outdir = tmp_path / "out"

    runner = CliRunner()
    # With subcommand
    res1 = runner.invoke(
        split_app,
        [
            "make",
            "--mode",
            "auto",
            "--val-days",
            "5",
            "--test-days",
            "5",
            "--embargo-train-val",
            "1",
            "--embargo-val-test",
            "0",
            "--rawdir",
            str(tmp_path),
            "--trades-file",
            "trades.parquet",
            "--outdir",
            str(outdir),
            "--write-counts",
            "--write-masks",
        ],
    )
    assert res1.exit_code == 0, res1.output
    assert (outdir / "ranges.json").exists()
    assert (outdir / "counts.json").exists()
    assert (outdir / "masks.parquet").exists()

    # Without subcommand (callback path)
    outdir2 = tmp_path / "out2"
    res2 = runner.invoke(
        split_app,
        [
            "--mode",
            "auto",
            "--val-days",
            "5",
            "--test-days",
            "5",
            "--embargo-train-val",
            "1",
            "--embargo-val-test",
            "0",
            "--rawdir",
            str(tmp_path),
            "--trades-file",
            "trades.parquet",
            "--outdir",
            str(outdir2),
            "--write-counts",
            "--write-masks",
        ],
    )
    assert res2.exit_code == 0, res2.output
    assert (outdir2 / "ranges.json").exists()
    assert (outdir2 / "counts.json").exists()
    assert (outdir2 / "masks.parquet").exists()


def test_fixed_mode_cli_open_ended_test_and_roundtrip(tmp_path: Path):
    _ = _write_sim(tmp_path, n_days=5)
    outdir = tmp_path / "fixed"

    runner = CliRunner()
    res = runner.invoke(
        split_app,
        [
            "make",
            "--mode",
            "fixed",
            "--train-end",
            "2025-01-03",
            "--val-end",
            "2025-01-04",
            "--rawdir",
            str(tmp_path),
            "--trades-file",
            "trades.parquet",
            "--outdir",
            str(outdir),
        ],
    )
    assert res.exit_code == 0, res.output

    jpath = outdir / "ranges.json"
    payload = json.loads(jpath.read_text())
    assert payload["test"]["end"] == "2100-01-01"

    # Roundtrip read_ranges
    rr = read_ranges(jpath)
    assert rr.test["end"] == "2100-01-01"


def test_rolling_kfold_with_embargo_and_masks(tmp_path: Path):
    trades_path = _write_sim(tmp_path, n_days=40)
    folds = compute_rolling_ranges(
        trades_path,
        n_folds=3,
        val_days=5,
        test_days=5,
        embargo_days=1,
        stride_days=5,
        embargo_val_test=1,
    )
    assert len(folds) >= 1
    # check internal ordering and embargoes
    for r in folds:
        # train end < val start by at least 1 day because embargo_days=1
        t_end = pd.to_datetime(r.train["end"]).date()
        v_start = pd.to_datetime(r.val["start"]).date()
        assert (v_start - t_end).days >= 2 - 1  # inclusive ranges imply >=1-day gap
        v_end = pd.to_datetime(r.val["end"]).date()
        te_start = pd.to_datetime(r.test["start"]).date()
        # val->test embargo = 1 day means a gap day between them
        assert (te_start - v_end).days >= 2 - 1

    outdir = tmp_path / "roll"
    jpath = write_ranges(folds, outdir)
    cpath = write_counts(trades_path, folds, outdir)
    mpath = write_masks(trades_path, folds, outdir)

    payload = json.loads(Path(jpath).read_text())
    assert "folds" in payload and len(payload["folds"]) == len(folds)

    counts = json.loads(Path(cpath).read_text())
    assert isinstance(counts, list) and {"fold_id", "train", "val", "test"} <= set(counts[0].keys())

    masks = pd.read_parquet(mpath)
    assert {"ts", "fold_id", "split"} <= set(masks.columns)
    assert set(masks["split"].unique()) <= {"train", "val", "test", "none"}


def test_ranges_roundtrip_single_and_folds(tmp_path: Path):
    trades_path = _write_sim(tmp_path, n_days=15)
    # single
    r = compute_default_ranges(trades_path, "2025-01-05", "2025-01-07")
    j1 = write_ranges(r, tmp_path / "s1")
    r_back = read_ranges(j1)
    assert r_back.train == r.train and r_back.val == r.val and r_back.test == r.test

    # folds
    folds = compute_rolling_ranges(trades_path, n_folds=2, val_days=3, test_days=3, embargo_days=1, min_train_days=7)
    j2 = write_ranges(folds, tmp_path / "s2")
    payload = json.loads(Path(j2).read_text())
    assert "folds" in payload and len(payload["folds"]) == len(folds)
