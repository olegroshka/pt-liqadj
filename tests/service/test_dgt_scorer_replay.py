import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ptliq.service.scoring import DGTScorer


def test_dgt_scorer_replay_model_artifacts(tmp_path: Path):
    # Use an existing trained model artifacts directory
    outdir = Path("models/dgt0")
    assert outdir.exists(), "Expected models/dgt0 with training artifacts to exist"

    # Initialize scorer from directory
    scorer = DGTScorer.from_dir(outdir, device="cpu")

    # Determine samples parquet via meta
    meta = json.loads((outdir / "mvdgt_meta.json").read_text())
    samples_path = meta.get("files", {}).get("samples")
    assert samples_path and Path(samples_path).exists(), "samples.parquet not found from meta"

    samp = pd.read_parquet(samples_path)
    assert "split" in samp.columns, "samples.parquet must contain split column"

    # pick a small deterministic batch from train split
    df = samp[samp["split"] == "train"].sample(n=16, random_state=17)

    # if market_index provided, construct mapping row_idx -> date
    mkt_idx_path = meta.get("files", {}).get("market_index")
    r2d = None
    if mkt_idx_path and Path(mkt_idx_path).exists():
        idx_df = pd.read_parquet(mkt_idx_path)
        r2d = {int(r.row_idx): pd.to_datetime(r.asof_date).normalize() for r in idx_df.itertuples(index=False)}

    # build user-facing rows
    rows = []
    for r in df.itertuples(index=False):
        side = "buy" if float(getattr(r, "side_sign", 0.0)) >= 0 else "sell"
        # invert log1p for human-friendly size
        size = float(np.expm1(abs(float(getattr(r, "log_size", 0.0)))))
        row = {"isin": str(r.isin), "side": side, "size": size}
        if r2d is not None:
            row["asof_date"] = r2d.get(int(r.date_idx))
        rows.append(row)

    # scorer predictions
    y_scorer = scorer.score_many(rows)

    # manual forward pass using loaded artifacts
    with torch.no_grad():
        anchor_idx = torch.tensor([scorer._isin_to_node[rr["isin"]] for rr in rows], dtype=torch.long)
        side_sign = torch.tensor([1.0 if rr["side"].lower().startswith("b") else -1.0 for rr in rows], dtype=torch.float32)
        log_size = torch.tensor([np.log1p(rr["size"]) for rr in rows], dtype=torch.float32)

        trade_raw = torch.stack([side_sign, log_size], dim=1)
        denom = torch.where(scorer._scaler_std <= 0, torch.ones_like(scorer._scaler_std), scorer._scaler_std)
        trade = (trade_raw - scorer._scaler_mean) / denom
        trade = torch.nan_to_num(trade, nan=0.0, posinf=0.0, neginf=0.0)

        market_feat = None
        if scorer.mkt_ctx is not None:
            idxs = []
            if getattr(scorer, "_mkt_dates", None) is not None and getattr(scorer, "_mkt_idxs", None) is not None:
                dates = scorer._mkt_dates
                idx_arr = scorer._mkt_idxs
                for rr in rows:
                    ts = pd.to_datetime(rr.get("asof_date")).normalize() if rr.get("asof_date") is not None else None
                    if ts is None:
                        pos = len(idx_arr) - 1
                    else:
                        pos = int(dates.searchsorted(ts, side="right") - 1)
                        if pos < 0:
                            pos = 0
                        if pos >= len(idx_arr):
                            pos = len(idx_arr) - 1
                    idxs.append(int(idx_arr[pos]))
            else:
                idxs = [int(scorer.mkt_ctx["mkt_feat"].size(0) - 1)] * len(rows)
            market_feat = scorer.mkt_ctx["mkt_feat"].index_select(0, torch.tensor(idxs, dtype=torch.long))

        y_manual = scorer.model(
            scorer.x, anchor_idx=anchor_idx, market_feat=market_feat,
            pf_gid=None, port_ctx=scorer.port_ctx, trade_feat=trade
        ).detach().cpu().numpy().reshape(-1)

    assert np.allclose(y_scorer, y_manual, atol=1e-5), (
        f"Scorer mismatch: max abs diff={np.max(np.abs(y_scorer - y_manual))}"
    )
