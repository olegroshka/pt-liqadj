import math
import numpy as np
import pandas as pd
import pytest
import torch

from pathlib import Path

from ptliq.training.mvdgt_loop import (
    MVDGTModelConfig, MVDGTTrainConfig, train_mvdgt
)
from ptliq.scoring import DGTScorer

try:
    from tests.test_mvdgt_e2e import _generate_toy_world
    HAVE_GENERATOR = True
except Exception:
    HAVE_GENERATOR = False


@pytest.mark.skipif(not HAVE_GENERATOR, reason="toy-world generator not importable")
@pytest.mark.e2e
def test_diag_pf_path_delta_sign(tmp_path: Path, capfd):
    """
    Toggle pf_gate ON and OFF to measure the *pure* portfolio residual contribution Δ_pf = y(ON) - y(OFF)
    on a basket constructed to produce a negative prototype (by generator design).
    This avoids confounding from graph / market / trade parts of the head.
    """
    work, pyg_dir, out, ctx = _generate_toy_world(tmp_path / "toy_pf_delta", seed=52)
    cfg = MVDGTTrainConfig(
        workdir=work, pyg_dir=pyg_dir, outdir=out,
        epochs=20, lr=5e-3, weight_decay=1e-4, batch_size=128,
        seed=23, device="cpu", enable_tb=False, enable_tqdm=False,
        model=MVDGTModelConfig(hidden=64, heads=2, dropout=0.10, trade_dim=2, use_portfolio=True),
    )
    train_mvdgt(cfg)
    scorer = DGTScorer.from_dir(out)

    X, U, portfolios = ctx["X"], ctx["U"], ctx["portfolios"]
    assert U is not None and portfolios is not None and len(portfolios) >= 1
    # Pick anchor with max projection and two most-negative co-items in the same group
    best = None
    for g, members in enumerate(portfolios):
        if len(members) < 3: continue
        members = list(map(int, members))
        u = U[g]; proj = np.dot(X[members], u)
        a_local = int(np.argmax(proj)); anchor = int(members[a_local])
        order = np.argsort(proj).tolist()
        co = [int(members[i]) for i in order if int(members[i]) != anchor][:2]
        if len(co) >= 2:
            neg_mean = float(proj[order[0]] + proj[order[1]]) / 2.0
            sep = float(proj[a_local] - neg_mean)
            if (best is None) or (sep > best["sep"]):
                best = dict(g=g, anchor=anchor, co=tuple(co), sep=sep)
    assert best is not None
    g = int(best["g"]); anchor_nid = int(best["anchor"]); co1, co2 = map(int, best["co"])

    # Build baskets (equal/opposite co-items => signed ~ 0, abs > 0)
    asof = str(pd.Timestamp(ctx["market_dates"].iloc[-1].asof_date).date())
    base_size = 1.5e5; co_size = 2.0e5
    def _isin(nid): return str(ctx["nodes_df"].iloc[nid]["isin"])
    pid = f"PFDELTA_G{g}"

    def basket(side):
        P0 = [{"portfolio_id": pid, "isin": _isin(anchor_nid), "side": side, "size": base_size, "asof_date": asof}]
        Pdrag = P0 + [
            {"portfolio_id": pid, "isin": _isin(co1), "side": "BUY",  "size": co_size, "asof_date": asof},
            {"portfolio_id": pid, "isin": _isin(co2), "side": "SELL", "size": co_size, "asof_date": asof},
        ]
        return P0, Pdrag

    # Read model and flip pf_gate
    ckpt = torch.load(out / "ckpt.pt", map_location="cpu")
    model = scorer.model
    base_gate = float(torch.sigmoid(getattr(model, "pf_gate")).item()) if hasattr(model, "pf_gate") else None
    print(f"[DIAG] pf_gate(base-sigmoid)={base_gate}")

    # Helper: run scorer with current gate
    def score_rows(rows):
        return float(scorer.score_many(rows)[0])

    for side in ["BUY", "SELL"]:
        P0, Pdrag = basket(side)
        # ON
        if hasattr(model, "pf_gate"):
            with torch.no_grad(): model.pf_gate.copy_(torch.tensor(10.0))  # ~1.0 sigmoid
        y0_on, yD_on = score_rows(P0), score_rows(Pdrag)
        # OFF
        if hasattr(model, "pf_gate"):
            with torch.no_grad(): model.pf_gate.copy_(torch.tensor(-10.0))  # ~0.0 sigmoid
        y0_off, yD_off = score_rows(P0), score_rows(Pdrag)

        delta_on  = yD_on  - y0_on
        delta_off = yD_off - y0_off
        delta_pf  = (yD_on - yD_off) - (y0_on - y0_off)  # isolates the pf residual effect between baskets

        print(f"[DIAG] side={side}  Δ_on={delta_on:.6f}  Δ_off={delta_off:.6f}  Δ_pf={delta_pf:.6f}")

    # Restore gate
    if hasattr(model, "pf_gate"):
        with torch.no_grad(): model.pf_gate.copy_(torch.tensor(math.log(base_gate/(1-base_gate))) if (base_gate is not None and 0<base_gate<1) else torch.tensor(0.0))
