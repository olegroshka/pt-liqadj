from __future__ import annotations
from pathlib import Path
import json

import pandas as pd

from ptliq.web.sample_payload import (
    load_samples_df,
    choose_portfolio_id,
    make_payload_for_portfolio,
    PayloadOptions,
    payload_to_compact_json,
    prepare_samples_in_workdir,
)


def _make_df(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "portfolio_id": ["P1", "P1", "P2"],
            "isin": ["US1", "US2", "US3"],
            "side": ["buy", "sell", "buy"],
            "size": [100_000, 50_000, 200_000],
        }
    )
    p = tmp_path / "samples.parquet"
    df.to_parquet(p, index=False)
    return p


def test_load_choose_and_make_payload(tmp_path: Path):
    p = _make_df(tmp_path)
    df = load_samples_df(source=p)

    # choose first portfolio by appearance
    pid = choose_portfolio_id(df)
    assert pid == "P1"

    # build payload for P1, limit to 2 rows, with deterministic jitter
    opts = PayloadOptions(n_rows=2, size_jitter=0.10, random_state=42)
    payload = make_payload_for_portfolio(df, pid, opts)

    assert "rows" in payload and isinstance(payload["rows"], list)
    rows = payload["rows"]
    assert len(rows) == 2
    # all rows should belong to the chosen portfolio
    assert all(r["portfolio_id"] == "P1" for r in rows)
    # check sides and isins preserved (lowercased side)
    assert rows[0]["isin"] == "US1" and rows[0]["side"] == "buy"
    assert rows[1]["isin"] == "US2" and rows[1]["side"] == "sell"
    # sizes should be close to originals within +/-10%
    orig = [100_000, 50_000]
    for r, o in zip(rows, orig):
        s = int(r["size"])  # jittered int
        assert int(0.9 * o) <= s <= int(1.1 * o)

    # Compact JSON formatting has one line per row entry
    js = payload_to_compact_json(payload)
    lines = js.splitlines()
    # two row lines expected (indented 4 spaces + { ... })
    row_lines = [
        ln for ln in lines
        if ln.lstrip().startswith("{") and (ln.rstrip().endswith("}") or ln.rstrip().endswith("},"))
    ]
    assert len(row_lines) == len(rows)
    # also ensure it is valid JSON
    js_obj = json.loads(js)
    assert js_obj == payload


def test_prepare_samples_in_workdir_copies(tmp_path: Path):
    src = _make_df(tmp_path)
    work = tmp_path / "work"
    out_path = prepare_samples_in_workdir(src, work)
    assert out_path.exists()
    # read back and compare basic shape
    df = pd.read_parquet(out_path)
    assert set(["portfolio_id", "isin", "side", "size"]).issubset(df.columns)


def test_load_legacy_schema_and_derive_side_and_size(tmp_path: Path):
    # Create legacy-format samples with side_sign and log_size only
    df_legacy = pd.DataFrame(
        {
            "portfolio_id": ["P1", "P1", "P2", "P3"],
            "isin": ["US1", "US2", "US3", "US4"],
            "side_sign": [1.0, -1.0, 0.0, float("nan")],
            "log_size": [11.512925, 10.819778, 0.0, 5.0],  # ~100k, ~50k, 1, ~148
        }
    )
    p = tmp_path / "samples_legacy.parquet"
    df_legacy.to_parquet(p, index=False)

    df = load_samples_df(source=p)
    # Loader should have derived 'side' and 'size'
    assert set(["portfolio_id", "isin", "side", "size"]).issubset(df.columns)
    # Check side mapping
    sides = df.loc[:3, "side"].tolist()
    assert sides[0] == "buy"
    assert sides[1] == "sell"
    assert sides[2] is None or pd.isna(sides[2])
    assert sides[3] is None or pd.isna(sides[3])
    # Check size mapping: exp(log_size) rounded and at least 1
    sizes = df.loc[:3, "size"].tolist()
    assert abs(int(sizes[0]) - 100000) <= 1
    assert abs(int(sizes[1]) - 50000) <= 1
    assert int(sizes[2]) >= 1
    assert int(sizes[3]) >= 1

    # Payload building should work off derived columns
    pid = choose_portfolio_id(df)
    payload = make_payload_for_portfolio(df, pid, PayloadOptions(n_rows=2, size_jitter=0.0, random_state=1))
    assert "rows" in payload and len(payload["rows"]) >= 1
