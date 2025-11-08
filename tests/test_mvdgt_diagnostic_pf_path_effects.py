# tests/test_mvdgt_diagnostic_pf_path_effects.py
import numpy as np
import pandas as pd
import torch
import pytest

from pathlib import Path
from ptliq.training.mvdgt_loop import MVDGTModelConfig, MVDGTTrainConfig, train_mvdgt
from tests.test_mvdgt_e2e import _generate_toy_world, _make_isin, DGTScorer

@pytest.mark.skipif("tests.test_mvdgt_e2e" is None, reason="toy-world generator helpers unavailable")
@pytest.mark.e2e
def test_diag_pf_path_response_to_pos_vs_neg_projection(tmp_path: Path, capfd):
    work, pyg_dir, out, ctx = _generate_toy_world(tmp_path / "toy_pf_dir", seed=101)
    cfg = MVDGTTrainConfig(
        workdir=work, pyg_dir=pyg_dir, outdir=out,
        epochs=20, lr=5e-3, weight_decay=1e-4, batch_size=128,
        seed=23, device="cpu", enable_tb=False, enable_tqdm=False,
        model=MVDGTModelConfig(hidden=64, heads=2, dropout=0.10, trade_dim=2, use_portfolio=True),
    )
    _ = train_mvdgt(cfg)
    scorer = DGTScorer.from_dir(out)

    X = ctx["X"]; U = ctx["U"]; portfolios = ctx["portfolios"]
    assert U is not None and portfolios is not None
    # pick one group with at least 3 names
    pick = None
    for g, members in enumerate(portfolios):
        if len(members) < 3:
            continue
        members = list(map(int, members))
        u = U[g]
        proj = np.dot(X[members], u)
        anchor = int(members[int(np.argmax(proj))])
        neg = int(members[int(np.argmin(proj))])
        pos = anchor  # "pos co-item" ~ same direction as anchor; if equal to anchor, pick next best
        # try second-highest if needed
        second = int(members[np.argsort(proj)[-2]])
        pos = second if second != anchor else second
        pick = (g, anchor, pos, neg); break
    assert pick is not None
    g, anchor_nid, pos_nid, neg_nid = pick

    asof = str(pd.Timestamp(ctx["market_dates"].iloc[-1].asof_date).date())
    base = 1.5e5; co = 2.0e5

    def basket(side, pid, co_isin, co_side):
        return [
            {"portfolio_id": pid, "pf_gid": g, "isin": _make_isin(anchor_nid), "side": side, "size": base, "asof_date": asof},
            {"portfolio_id": pid, "pf_gid": g, "isin": co_isin, "side": co_side, "size": co, "asof_date": asof},
        ]

    # a) negative projection co-item pair (BUY/SELL is chosen to emphasize absolute path)
    P_neg = basket("BUY", "NEG", _make_isin(neg_nid), "BUY")
    y0 = float(scorer.score_many(P_neg[:1])[0]); yD_neg = float(scorer.score_many(P_neg)[0])
    delta_neg = yD_neg - y0

    # b) positive projection co-item
    P_pos = basket("BUY", "POS", _make_isin(pos_nid), "BUY")
    yD_pos = float(scorer.score_many(P_pos)[0])
    delta_pos = yD_pos - y0

    # Gate off (for reference)
    scorer.model.pf_gate.data.fill_(-10.0)  # sigmoid ~ 0
    y0_off = float(scorer.score_many(P_neg[:1])[0]); yD_neg_off = float(scorer.score_many(P_neg)[0])
    yD_pos_off = float(scorer.score_many(P_pos)[0])

    print(f"[DIAG] pf_gate(sigmoid) on = {float(torch.sigmoid(scorer.model.pf_gate).item()):.6f}")
    print(f"[DIAG] Δ_neg(on)={delta_neg:.6f}   Δ_pos(on)={delta_pos:.6f}")
    print(f"[DIAG] Δ_neg(off)={yD_neg_off - y0_off:.6f}   Δ_pos(off)={yD_pos_off - y0_off:.6f}")

    # No assert here; this is purely directional debugging info.
