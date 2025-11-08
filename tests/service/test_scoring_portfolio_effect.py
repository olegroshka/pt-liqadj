from __future__ import annotations
import json
from pathlib import Path
import math
import numpy as np
import pandas as pd
import pytest

from ptliq.service.scoring import DGTScorer


def _key(row: dict) -> tuple:
    # permutation-invariant key for a line item
    isin = str(row.get("isin", "")).strip()
    side = str(row.get("side", "")).strip().upper()
    try:
        sz = float(row.get("size", 0.0))
    except Exception:
        sz = 0.0
    # use log1p(abs(size)) to avoid scale issues; round to stable precision
    lsz = round(math.log1p(abs(sz)), 6)
    return (isin, side, lsz)


def _load_isins_from_scorer(scorer) -> list[str]:
    try:
        return sorted([str(k) for k in scorer._isin_to_node.keys()])
    except Exception:
        return []


def _pick_asof_date(workdir: Path) -> str | None:
    try:
        meta = json.loads((workdir / "mvdgt_meta.json").read_text())
        mkt_index = meta["files"].get("market_index")
        if mkt_index and Path(mkt_index).exists():
            idx_df = pd.read_parquet(mkt_index)
            idx_df["asof_date"] = pd.to_datetime(idx_df["asof_date"]).dt.normalize()
            return str(pd.Timestamp(idx_df["asof_date"].max()).date())
    except Exception:
        pass
    return None


@pytest.mark.parametrize("model_dir", ["models/dgt0", "models/dgt_demo"], ids=["dgt0", "demo"])
def test_portfolio_composition_changes_affect_common_item(model_dir: str):
    root = Path(__file__).resolve().parents[2]
    workdir = root / model_dir
    if not (workdir / "ckpt.pt").exists():
        pytest.skip(f"Artifacts not found at {workdir}")

    scorer = DGTScorer.from_dir(workdir)
    isins = _load_isins_from_scorer(scorer)
    assert len(isins) >= 2, "Insufficient nodes for constructing baskets"

    asof = _pick_asof_date(workdir) or (str(pd.Timestamp.max.normalize().date()) if hasattr(pd, 'Timestamp') else None)
    common = {"isin": isins[0], "side": "BUY", "size": 200_000}
    if asof:
        common["asof_date"] = asof

    P1 = [
        {"portfolio_id": "PF_A", **common},
        {"portfolio_id": "PF_A", "isin": isins[1], "side": "BUY",  "size": 100_000},
        {"portfolio_id": "PF_A", "isin": isins[2], "side": "SELL", "size": 300_000},
    ]
    P2 = [
        {"portfolio_id": "PF_B", **common},
        {"portfolio_id": "PF_B", "isin": isins[3], "side": "SELL", "size": 500_000},
        {"portfolio_id": "PF_B", "isin": isins[4], "side": "SELL", "size": 200_000},
        {"portfolio_id": "PF_B", "isin": isins[5], "side": "BUY",  "size": 150_000},
    ]

    y1 = scorer.score_many(P1)
    y2 = scorer.score_many(P2)

    # Compare the score for the common first row in both baskets
    assert len(y1) == len(P1) and len(y2) == len(P2)
    delta = float(abs(float(y1[0]) - float(y2[0])))
    # The effect can be modest depending on checkpoint; use a small threshold
    assert delta > 1e-4, f"Portfolio conditioning had no visible effect on common item: Δ={delta:.6f}"


@pytest.mark.parametrize("model_dir", ["models/dgt_demo"], ids=["dg_demo"])  # one artifact is enough for invariance properties
def test_permutation_and_id_invariance_predictions(model_dir: str):
    root = Path(__file__).resolve().parents[2]
    workdir = root / model_dir
    if not (workdir / "ckpt.pt").exists():
        pytest.skip(f"Artifacts not found at {workdir}")

    scorer = DGTScorer.from_dir(workdir)
    isins = _load_isins_from_scorer(scorer)
    assert len(isins) >= 4, "Insufficient nodes for constructing a basket"

    basket = [
        {"portfolio_id": "ABC", "isin": isins[0], "side": "BUY",  "size": 100_000},
        {"portfolio_id": "ABC", "isin": isins[1], "side": "SELL", "size": 200_000},
        {"portfolio_id": "ABC", "isin": isins[2], "side": "BUY",  "size": 300_000},
        {"portfolio_id": "ABC", "isin": isins[3], "side": "SELL", "size": 400_000},
    ]
    # permute order
    basket_perm = [basket[i] for i in [2, 0, 3, 1]]
    # change only the portfolio_id label
    basket_id2 = [{**r, "portfolio_id": "XYZ"} for r in basket]

    y = scorer.score_many(basket)
    y_p = scorer.score_many(basket_perm)
    y_id = scorer.score_many(basket_id2)

    # Build key->score maps
    kv = {_key(r): float(y[i]) for i, r in enumerate(basket)}
    kv_p = {_key(r): float(y_p[i]) for i, r in enumerate(basket_perm)}
    kv_id = {_key(r): float(y_id[i]) for i, r in enumerate(basket_id2)}

    # Permutation invariance: scores per keyed item must match exactly
    for k, v in kv.items():
        assert k in kv_p, f"Missing key after permutation: {k}"
        assert np.isclose(v, kv_p[k], atol=1e-5, rtol=0.0), f"Permutation changed score for {k}: {v} vs {kv_p[k]}"

    # Portfolio-ID invariance: changing string label must not change weights, hence same scores
    for k, v in kv.items():
        assert k in kv_id, f"Missing key after id change: {k}"
        # Allow tiny float noise due to non-determinism; keep strict but realistic tolerance
        assert np.isclose(v, kv_id[k], atol=1e-5, rtol=0.0), f"Portfolio ID label changed score for {k}: {v} vs {kv_id[k]}"
