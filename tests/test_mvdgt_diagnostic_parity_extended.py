# tests/test_mvdgt_diagnostic_parity_extended.py
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from ptliq.training.mvdgt_loop import MVDGTModelConfig, MVDGTTrainConfig, train_mvdgt
from tests.test_mvdgt_e2e import _generate_toy_world, _make_isin, DGTScorer, _direct_predict

@pytest.mark.e2e
def test_diag_parity_multirow_with_and_without_pf_gid(tmp_path: Path, capfd):
    work, pyg_dir, out, ctx = _generate_toy_world(tmp_path / "toy_parity_ext", seed=51)
    cfg = MVDGTTrainConfig(
        workdir=work, pyg_dir=pyg_dir, outdir=out,
        epochs=15, lr=5e-3, weight_decay=1e-4, batch_size=128,
        seed=19, device="cpu", enable_tb=False, enable_tqdm=False,
        model=MVDGTModelConfig(hidden=64, heads=2, dropout=0.10, trade_dim=2, use_portfolio=True),
    )
    _ = train_mvdgt(cfg)
    scorer = DGTScorer.from_dir(out)

    # Build a small warm batch like your extended-e2e does
    samp = pd.read_parquet(work / "samples.parquet")
    warm_tr = samp[(samp["split"]=="train") & (samp["pf_gid"]>=0)]
    rows = []
    for r in warm_tr.sample(6, random_state=7).itertuples(index=False):
        rows.append({
            "portfolio_id": f"PF-{int(r.pf_gid)}",
            "pf_gid": int(r.pf_gid),
            "isin": _make_isin(int(r.node_id)),
            "side": "BUY" if float(r.side_sign)>0 else "SELL",
            "size": float(np.expm1(float(r.log_size))),
            "asof_date": str(pd.Timestamp(ctx["market_dates"].iloc[int(r.date_idx)].asof_date).date()),
        })

    preds_scorer = scorer.score_many(rows)
    preds_direct = _direct_predict(out, rows)

    # Now remove pf_gid to force scorer to fall back to portfolio_id grouping only
    rows_no_gid = [{k:v for k,v in rr.items() if k!="pf_gid"} for rr in rows]
    preds_scorer_nogid = scorer.score_many(rows_no_gid)
    preds_direct_same  = _direct_predict(out, rows_no_gid)

    print(f"[DIAG] max|scorer-direct| (with gid)    = {float(np.max(np.abs(preds_scorer - preds_direct))):.9f}")
    print(f"[DIAG] max|scorer-direct| (without gid) = {float(np.max(np.abs(preds_scorer_nogid - preds_direct_same))):.9f}")
    print(f"[DIAG] preds_scorer(with gid)    = {preds_scorer.tolist()}")
    print(f"[DIAG] preds_scorer(without gid) = {preds_scorer_nogid.tolist()}")

    # No assert; intended to show where parity breaks when grouping signals differ.
