import json
import math
import numpy as np
import pandas as pd
import pytest
import torch

from pathlib import Path

from ptliq.model.mv_dgt import MultiViewDGT
from ptliq.training.mvdgt_loop import (
    MVDGTModelConfig, MVDGTTrainConfig, train_mvdgt
)
from ptliq.scoring import DGTScorer

# Reuse the toy-world builder from your existing e2e test if available.
try:
    from tests.test_mvdgt_e2e import _generate_toy_world, _direct_predict
    HAVE_GENERATOR = True
except Exception:
    HAVE_GENERATOR = False

@pytest.mark.skipif(not HAVE_GENERATOR, reason="toy-world generator not importable")
@pytest.mark.e2e
def test_diag_parity_multirow_breakdown(tmp_path: Path, capfd):
    """
    Build a warm basket with 3 rows and print step-by-step:
      - trade feature tensors (scaler usage)
      - market z-scoring + sign flip
      - constructed portfolio ctx (nodes, abs/signed weights, lens)
      - y_scorer vs y_direct_and_portctx_equalized
    Goal: pinpoint where the scorer vs direct difference creeps in on multi-row input.
    """
    work, pyg_dir, out, ctx = _generate_toy_world(tmp_path / "toy_parity_diag", seed=44)
    cfg = MVDGTTrainConfig(
        workdir=work, pyg_dir=pyg_dir, outdir=out,
        epochs=15, lr=5e-3, weight_decay=1e-4, batch_size=128,
        seed=19, device="cpu", enable_tb=False, enable_tqdm=False,
        model=MVDGTModelConfig(hidden=64, heads=2, dropout=0.10, trade_dim=2, use_portfolio=True),
    )
    train_mvdgt(cfg)

    scorer = DGTScorer.from_dir(out)

    # Pick one warm group & day with 3 rows
    samp = pd.read_parquet(work / "samples.parquet")
    warm = samp[(samp["split"] == "train") & (samp["pf_gid"] >= 0)].copy()
    assert len(warm) >= 3
    rows = [warm.sample(1, random_state=rs).iloc[0] for rs in (10, 11, 12)]
    nid2isin = {int(r.node_id): str(r.isin) for r in ctx["nodes_df"].itertuples(index=False)}
    asof = str(pd.Timestamp(ctx["market_dates"].iloc[-1].asof_date).date())

    # Build a 3-row basket in the *same* portfolio_id / pf_gid
    gid = int(rows[0].pf_gid)
    pid = f"PARITY_G{gid}"
    req = []
    for r in rows:
        side = "BUY" if float(r.side_sign) > 0 else "SELL"
        size = float(math.expm1(float(r.log_size)))
        req.append({
            "portfolio_id": pid, "pf_gid": gid,
            "isin": nid2isin[int(r.node_id)], "side": side, "size": size, "asof_date": asof,
        })

    y_scorer = scorer.score_many(req)
    print(f"[DIAG] scorer preds: {np.round(y_scorer, 6).tolist()}")

    # Direct path used in e2e helper (will likely differ on multi-row)
    y_direct = _direct_predict(out, req)
    print(f"[DIAG] direct preds: {np.round(y_direct, 6).tolist()}")

    # Try to *equalize* portfolio ctx: build a single port_ctx out of req (anchor-inclusive),
    # then feed that port_ctx to the model (model applies strict LOO internally).
    from ptliq.model.mv_dgt import compute_samplewise_portfolio_vectors_loo
    from ptliq.training.mvdgt_loop import _load_pyg_and_view_masks, copy_and_augment_meta, _standardize_trade_batch, _standardize_market_batch, _prepare_market_preproc_tensors
    import torch

    # Recreate state similar to your "_rebuild_state_for_manual_forward"
    meta = json.loads((out / "mvdgt_meta.json").read_text())
    device = torch.device("cpu")
    x, edge_index, edge_weight, view_masks = _load_pyg_and_view_masks(meta, work, device, out, logger=_DummyLogger())
    ckpt = torch.load(out / "ckpt.pt", map_location=device)
    mcfg = MVDGTModelConfig(**ckpt["model_config"])
    model = MultiViewDGT(
        x_dim=mcfg.x_dim, hidden=mcfg.hidden, heads=mcfg.heads, dropout=mcfg.dropout,
        view_masks=view_masks, edge_index=edge_index, edge_weight=edge_weight,
        mkt_dim=(mcfg.mkt_dim or 0), use_portfolio=True, use_market=bool(mcfg.use_market), trade_dim=mcfg.trade_dim,
        view_names=mcfg.views,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Build tensors for the basket
    df = pd.DataFrame(req)
    # anchor_idx in request order
    nodes_df = ctx["nodes_df"]
    isin2nid = {str(r.isin): int(r.node_id) for r in nodes_df.itertuples(index=False)}
    anchor_idx = torch.tensor([isin2nid[str(s)] for s in df["isin"].tolist()], dtype=torch.long, device=device)
    pf_gid = torch.tensor(df["pf_gid"].astype(int).to_list(), dtype=torch.long, device=device)
    # trade
    scaler = json.loads((out / "scaler.json").read_text())
    scaler_mean_t = torch.tensor(scaler["mean"], dtype=torch.float32, device=device)
    scaler_std_t = torch.tensor([1.0 if (s is None or s <= 0.0) else float(s) for s in scaler["std"]], dtype=torch.float32, device=device)
    trade_raw = torch.stack([
        torch.tensor([+1.0 if s == "BUY" else -1.0 for s in df["side"]], dtype=torch.float32, device=device),
        torch.tensor([float(np.log1p(float(sz))) for sz in df["size"]], dtype=torch.float32, device=device),
    ], dim=1)
    trade = (trade_raw - scaler_mean_t) / torch.where(scaler_std_t <= 0, torch.ones_like(scaler_std_t), scaler_std_t)

    # market
    mkt_ctx = torch.load(Path(meta["files"]["market_context"]), map_location=device)
    samp = pd.read_parquet(work / "samples.parquet")
    m_mean_t, m_std_t, m_sign_t = _prepare_market_preproc_tensors(samp, mkt_ctx, device, out, logger=_DummyLogger())
    mi = ctx["market_dates"]
    di = [int(mi[mi["asof_date"] == np.datetime64(a)].index[0]) for a in df["asof_date"]]
    batch = {"date_idx": torch.tensor(di, dtype=torch.long)}
    mkt = _standardize_market_batch(mkt_ctx, batch, device, m_mean_t, m_std_t, m_sign_t)

    # Hand-built *inclusive* port_ctx from req
    # Equal-mass abs normalization inside each portfolio_id
    groups = {}
    for i, r in enumerate(req):
        groups.setdefault(r["portfolio_id"], []).append(i)

    port_nodes, w_abs, w_sgn, port_len = [], [], [], []
    for _, idxs in groups.items():
        sizes = torch.tensor([float(req[i]["size"]) for i in idxs], dtype=torch.float32)
        signs = torch.tensor([+1.0 if req[i]["side"] == "BUY" else -1.0 for i in idxs], dtype=torch.float32)
        a = torch.expm1(torch.log1p(sizes).clamp_min(0.0))  # same transformation as training’s dynamic ctx
        a = a / max(float(a.sum().item()), 1.0)
        s = signs * a
        port_len.append(len(idxs))
        port_nodes.extend(anchor_idx[idxs].tolist())
        w_abs.extend(a.tolist())
        w_sgn.extend(s.tolist())
    port_ctx = {
        "port_nodes_flat": torch.tensor(port_nodes, dtype=torch.long),
        "port_w_abs_flat": torch.tensor(w_abs, dtype=torch.float32),
        "port_w_signed_flat": torch.tensor(w_sgn, dtype=torch.float32),
        "port_len": torch.tensor(port_len, dtype=torch.long),
    }

    with torch.no_grad():
        y_dir_equalized = model(
            x, anchor_idx=anchor_idx, pf_gid=pf_gid, port_ctx=port_ctx,
            market_feat=mkt, trade_feat=trade, return_aux=False
        ).detach().cpu().numpy()

    print(f"[DIAG] direct (equalized port_ctx) preds: {np.round(y_dir_equalized, 6).tolist()}")
    d1 = np.max(np.abs(y_scorer - y_direct))
    d2 = np.max(np.abs(y_scorer - y_dir_equalized))
    print(f"[DIAG] max|scorer - direct|={float(d1):.6f}  |  max|scorer - direct_equalized|={float(d2):.6f}")

    # Only soft check here; hard asserts live in your e2e suite.
    assert d2 <= max(d1, 1e-3)

class _DummyLogger:
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass
