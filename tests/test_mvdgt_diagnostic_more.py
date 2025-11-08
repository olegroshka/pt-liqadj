# tests/test_mvdgt_diagnostic_more.py
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
import pytest

from ptliq.service.scoring import DGTScorer
from ptliq.training.mvdgt_loop import (
    train_mvdgt, MVDGTTrainConfig, MVDGTModelConfig
)

# Reuse your toy world helper exactly as in other tests
from tests.test_mvdgt_e2e import _generate_toy_world, _make_isin


def _as_date_str(ts) -> str:
    return str(pd.Timestamp(ts).date())


def _rebuild_state_for_manual_forward(outdir: Path) -> Dict[str, Any]:
    # minimal mirror of your other diag loader(s)
    ckpt = torch.load(outdir / "ckpt.pt", map_location="cpu")
    model_cfg = MVDGTModelConfig(**ckpt["model_config"])
    meta = json.loads((outdir / "mvdgt_meta.json").read_text())
    view_masks = torch.load(outdir / "view_masks.pt", map_location="cpu")
    data = torch.load(Path(meta["files"]["pyg_graph"]), map_location="cpu", weights_only=False)

    from ptliq.model.mv_dgt import MultiViewDGT
    device = torch.device("cpu")

    x = data.x.float().to(device)
    edge_index = data.edge_index.to(device)
    edge_weight = data.edge_weight.to(device) if hasattr(data, "edge_weight") else None
    vm = {k: v.to(device) for k, v in view_masks.items()}

    model = MultiViewDGT(
        x_dim=int(model_cfg.x_dim or x.size(1)),
        hidden=int(model_cfg.hidden),
        heads=int(model_cfg.heads),
        dropout=float(model_cfg.dropout),
        view_masks=vm,
        edge_index=edge_index,
        edge_weight=edge_weight,
        mkt_dim=int(model_cfg.mkt_dim or 0),
        use_portfolio=bool(model_cfg.use_portfolio),
        use_market=bool(model_cfg.use_market),
        trade_dim=int(model_cfg.trade_dim),
        view_names=list(getattr(model_cfg, "views", ["struct","port","corr_global","corr_local"])),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    # market ctx + index if available
    mkt_ctx = None
    mkt_lookup = None
    mkt_ctx_path = meta["files"].get("market_context")
    mkt_idx_path = meta["files"].get("market_index")
    if mkt_ctx_path and Path(mkt_ctx_path).exists():
        mkt_ctx = torch.load(mkt_ctx_path, map_location=device)
        if mkt_idx_path and Path(mkt_idx_path).exists():
            idx_df = pd.read_parquet(mkt_idx_path)
            idx_df["asof_date"] = pd.to_datetime(idx_df["asof_date"]).dt.normalize()
            mkt_lookup = {pd.Timestamp(r.asof_date): int(r.row_idx) for r in idx_df.itertuples(index=False)}

    nodes = pd.read_parquet(meta["files"]["graph_nodes"])
    isin_to_node = {str(r.isin): int(r.node_id) for r in nodes.itertuples(index=False)}

    # scaler + feature order
    fnames = json.loads((outdir / "feature_names.json").read_text())
    sc = json.loads((outdir / "scaler.json").read_text())
    mean = torch.tensor([float(sc["mean"][i]) for i in range(len(fnames))], dtype=torch.float32, device=device)
    std  = torch.tensor([float(sc["std"][i])  if (float(sc["std"][i]) > 0.0) else 1.0
                         for i in range(len(fnames))], dtype=torch.float32, device=device)

    # option: market preproc (might exist)
    pre = outdir / "market_preproc.json"
    mkt_mean = None; mkt_std = None; mkt_sign = 1.0
    if pre.exists():
        proto = json.loads(pre.read_text())
        mkt_mean = torch.tensor(proto.get("mean", []), dtype=torch.float32, device=device)
        mkt_std  = torch.tensor([v if (isinstance(v, (int,float)) and v>0) else 1.0 for v in proto.get("std", [])],
                                dtype=torch.float32, device=device)
        mkt_sign = float(proto.get("sign", 1.0))

    return dict(
        device=device, model=model, x=x, meta=meta, mkt_ctx=mkt_ctx, mkt_lookup=mkt_lookup,
        fnames=fnames, mean=mean, std=std, isin_to_node=isin_to_node, mkt_mean=mkt_mean,
        mkt_std=mkt_std, mkt_sign=mkt_sign
    )


def _vectorize_rows_for_state(state: Dict[str,Any], rows: List[Dict[str,Any]]):
    device = state["device"]
    fnames = state["fnames"]; mean=state["mean"]; std=state["std"]
    mkt_ctx = state["mkt_ctx"]; mkt_lookup = state["mkt_lookup"]
    mkt_mean = state["mkt_mean"]; mkt_std=state["mkt_std"]; mkt_sign=state["mkt_sign"]

    def _side_sign(v):
        if v is None: return 0.0
        s = str(v).strip().upper()
        return 1.0 if s in {"B","BUY","CBUY","TRUE","1"} else (-1.0 if s in {"S","SELL","CSELL","FALSE","0","-1"} else 0.0)

    def _log_size(v):
        try:
            f = float(v); return float(math.log1p(abs(f))) if f == f else 0.0
        except Exception:
            return 0.0

    def _to_date(v):
        if v is None: return None
        try:
            return pd.to_datetime(v, errors="coerce").normalize()
        except Exception:
            return None

    # anchors
    node_ids = []
    for r in rows:
        isin = str(r.get("isin","")).strip()
        nid = state["isin_to_node"].get(isin, None)
        if nid is None:
            raise ValueError(f"Unknown ISIN in diag: {isin}")
        node_ids.append(nid)
    anchor_idx = torch.as_tensor(node_ids, dtype=torch.long, device=device)

    # trade vectors
    side = [_side_sign(r.get("side")) for r in rows]
    lsz  = [_log_size(r.get("size")) for r in rows]
    n = len(rows)
    cols = []
    for i, name in enumerate(fnames):
        if name == "side_sign":
            cols.append(torch.as_tensor(side, dtype=torch.float32, device=device))
        elif name == "log_size":
            cols.append(torch.as_tensor(lsz, dtype=torch.float32, device=device))
        else:
            cols.append(torch.full((n,), float(mean[i].item()), dtype=torch.float32, device=device))
    raw = torch.stack(cols, dim=1) if cols else torch.zeros((n,0), dtype=torch.float32, device=device)
    trade = torch.nan_to_num((raw - mean) / torch.where(std <= 0, torch.ones_like(std), std),
                             nan=0.0, posinf=0.0, neginf=0.0)

    # market
    market_feat = None
    if mkt_ctx is not None:
        idxs = []
        last_idx = int(mkt_ctx["mkt_feat"].size(0) - 1)
        if mkt_lookup:
            items = sorted(mkt_lookup.items())
            dates_index = pd.to_datetime([k for k,_ in items])
            idx_arr = np.asarray([v for _,v in items], dtype=np.int64)
        else:
            dates_index = None; idx_arr = None
        for r in rows:
            ts = _to_date(r.get("asof_date"))
            if (ts is None) or (mkt_lookup is None):
                idxs.append(last_idx)
            elif ts in mkt_lookup:
                idxs.append(mkt_lookup[ts])
            else:
                if (dates_index is None) or (len(idx_arr)==0):
                    idxs.append(last_idx)
                else:
                    pos = int(dates_index.searchsorted(ts, side="right") - 1)
                    pos = max(0, min(pos, len(idx_arr)-1))
                    idxs.append(int(idx_arr[pos]))
        di = torch.as_tensor(idxs, dtype=torch.long, device=device)
        market_feat = mkt_ctx["mkt_feat"].index_select(0, di)
        if (mkt_mean is not None) and (mkt_std is not None):
            denom = torch.where(mkt_std <= 0, torch.ones_like(mkt_std), mkt_std)
            market_feat = (market_feat - mkt_mean) / denom
            market_feat = torch.nan_to_num(market_feat, nan=0.0, posinf=0.0, neginf=0.0)
            market_feat = market_feat * float(mkt_sign)
    return anchor_idx, trade, market_feat


def _build_manual_runtime_ctx(rows: List[Dict[str,Any]], node_ids: List[int], device: torch.device) -> dict:
    # mirrors scorer’s runtime ctx logic: abs ∝ |size|, signed ∝ sign*|size|
    def _side_sign(v):
        if v is None: return 0.0
        s = str(v).strip().upper()
        return 1.0 if s in {"B","BUY","CBUY","TRUE","1"} else (-1.0 if s in {"S","SELL","CSELL","FALSE","0","-1"} else 0.0)
    from collections import defaultdict
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        pid = r.get("portfolio_id")
        if pid is None:
            continue
        try:
            sz = float(r.get("size", 0.0))
        except Exception:
            sz = 0.0
        a = abs(sz); sgn = _side_sign(r.get("side")) * a
        groups[pid].append((i, int(node_ids[i]), a, sgn))
    if not groups:
        return {}
    port_nodes=[]; port_w_abs=[]; port_w_sgn=[]; port_len=[]
    for _, items in groups.items():
        abs_vec = np.array([a for (_,_,a,_) in items], dtype=np.float32)
        sgn_vec = np.array([s for (_,_,_,s) in items], dtype=np.float32)
        s = float(abs_vec.sum()); s = (s if s>0 and np.isfinite(s) else max(1.0, float(len(items))))
        w_abs = (abs_vec / s).astype(np.float32)
        w_sgn = (sgn_vec / s).astype(np.float32)
        port_len.append(len(items))
        for (_, nid, _a, _s), wA, wS in zip(items, w_abs, w_sgn):
            port_nodes.append(nid); port_w_abs.append(float(wA)); port_w_sgn.append(float(wS))
    return {
        "port_nodes_flat": torch.tensor(port_nodes, dtype=torch.long, device=device),
        "port_w_abs_flat": torch.tensor(port_w_abs, dtype=torch.float32, device=device),
        "port_w_signed_flat": torch.tensor(port_w_sgn, dtype=torch.float32, device=device),
        "port_len": torch.tensor(port_len, dtype=torch.long, device=device),
    }


@pytest.mark.e2e
def test_diag_pf_residual_breakdown_and_parity(tmp_path: Path, capfd):
    """
    Prints a per-row breakdown:
      - scorer vs direct
      - direct with NO portfolio context (pf_gid=-1) to isolate portfolio residual effect
      - direct with manually built runtime port_ctx (to confirm scorer parity on ctx construction)
    Helps diagnose the ~1e-2 parity drift seen in e2e when portfolio context is active.
    """
    work, pyg_dir, out, ctx = _generate_toy_world(tmp_path / "toy_par_diag", seed=71)
    cfg = MVDGTTrainConfig(
        workdir=work, pyg_dir=pyg_dir, outdir=out,
        epochs=25, lr=5e-3, weight_decay=1e-4, batch_size=128,
        seed=17, device="cpu", enable_tb=False, enable_tqdm=False,
        model=MVDGTModelConfig(hidden=64, heads=2, dropout=0.10, trade_dim=2, use_portfolio=True),
    )
    train_mvdgt(cfg)

    scorer = DGTScorer.from_dir(out)
    state = _rebuild_state_for_manual_forward(out)
    model = state["model"]; x=state["x"]; device = state["device"]

    # construct 3-row basket with a portfolio_id
    asof = str(pd.Timestamp(ctx["market_dates"].iloc[-1].asof_date).date())
    rows = [
        {"portfolio_id":"P1", "isin": _make_isin(0), "side":"BUY",  "size": 1.2e5, "asof_date":asof},
        {"portfolio_id":"P1", "isin": _make_isin(1), "side":"SELL", "size": 2.0e5, "asof_date":asof},
        {"portfolio_id":"P1", "isin": _make_isin(2), "side":"BUY",  "size": 2.0e5, "asof_date":asof},
    ]
    # scorer
    y_scorer = scorer.score_many(rows)

    # direct: need anchor_idx/trade/market
    anchor_idx, trade, market_feat = _vectorize_rows_for_state(state, rows)

    # build runtime ctx manually (to check parity w.r.t scorer)
    node_ids = anchor_idx.detach().cpu().tolist()
    port_ctx = _build_manual_runtime_ctx(rows, node_ids, device)
    pf_gid = torch.zeros((len(rows),), dtype=torch.long, device=device)  # single portfolio group 0

    with torch.no_grad():
        y_direct = model(x, anchor_idx=anchor_idx, market_feat=market_feat, pf_gid=pf_gid,
                         port_ctx=port_ctx, trade_feat=trade).cpu().numpy().astype(np.float32)

        # portfolio off (pf_gid=-1) to isolate residual contribution
        pf_gid_off = torch.full_like(pf_gid, -1)
        y_direct_off = model(x, anchor_idx=anchor_idx, market_feat=market_feat, pf_gid=pf_gid_off,
                             port_ctx=None, trade_feat=trade).cpu().numpy().astype(np.float32)

    print(f"[DIAG] preds scorer  : {np.round(y_scorer,6).tolist()}")
    print(f"[DIAG] preds direct  : {np.round(y_direct,6).tolist()}")
    print(f"[DIAG] preds no-pf   : {np.round(y_direct_off,6).tolist()}")
    print(f"[DIAG] max|scorer - direct| = {float(np.max(np.abs(y_scorer - y_direct))):.6f}")
    print(f"[DIAG] max|direct - no-pf|  = {float(np.max(np.abs(y_direct - y_direct_off))):.6f} (portfolio path magnitude)")

    # no assert → purely diagnostic


@pytest.mark.e2e
def test_diag_loo_vectors_when_anchor_present_and_absent(tmp_path: Path, capfd):
    """
    Pick a group, build two contexts:
      - ctx_loo: only co-items
      - ctx_self: anchor + co-items (self should be excluded by LOO)
    Print raw vectors V_abs, V_sgn and their difference.
    """
    work, pyg_dir, out, ctx = _generate_toy_world(tmp_path / "toy_vprobe", seed=77)
    cfg = MVDGTTrainConfig(
        workdir=work, pyg_dir=pyg_dir, outdir=out,
        epochs=10, lr=5e-3, weight_decay=1e-4, batch_size=128,
        seed=11, device="cpu", enable_tb=False, enable_tqdm=False,
        model=MVDGTModelConfig(hidden=32, heads=2, dropout=0.10, trade_dim=2, use_portfolio=True),
    )
    train_mvdgt(cfg)
    state = _rebuild_state_for_manual_forward(out)
    device = state["device"]; model = state["model"]; x=state["x"]

    # choose group with ≥3 members
    X, U, portfolios = ctx["X"], ctx["U"], ctx["portfolios"]
    pick = None
    for g, members in enumerate(portfolios):
        if len(members) < 3: continue
        u = U[g]; members = list(map(int, members))
        proj = np.dot(X[members], u)
        a = int(members[int(np.argmax(proj))])
        co = [int(members[i]) for i in np.argsort(proj).tolist() if int(members[i]) != a][:2]
        if len(co) >= 2:
            pick = (g, a, co[0], co[1]); break
    assert pick is not None
    g, anchor_nid, c1, c2 = pick

    anchor_idx = torch.tensor([anchor_nid], dtype=torch.long, device=device)
    pf_gid = torch.tensor([0], dtype=torch.long, device=device)  # single group

    def _ctx(nodes, w_abs, w_sgn):
        return {
            "port_nodes_flat": torch.tensor(nodes, dtype=torch.long, device=device),
            "port_w_abs_flat": torch.tensor(w_abs, dtype=torch.float32, device=device),
            "port_w_signed_flat": torch.tensor(w_sgn, dtype=torch.float32, device=device),
            "port_len": torch.tensor([len(nodes)], dtype=torch.long, device=device),
        }

    # co-items only vs self+co (weights arbitrary; LOO should remove the self anyway)
    ctx_loo  = _ctx([c1, c2], [0.5, 0.5], [0.0, 0.0])
    ctx_self = _ctx([anchor_nid, c1, c2], [0.4, 0.3, 0.3], [0.4, 0.3, -0.3])

    from ptliq.model.mv_dgt import compute_samplewise_portfolio_vectors_loo
    with torch.no_grad():
        V_abs_loo, V_sgn_loo = compute_samplewise_portfolio_vectors_loo(model._encode_nodes(x), anchor_idx, pf_gid, ctx_loo,  l2_normalize=False)
        V_abs_self, V_sgn_self = compute_samplewise_portfolio_vectors_loo(model._encode_nodes(x), anchor_idx, pf_gid, ctx_self, l2_normalize=False)

    print(f"[DIAG] V_abs (loo )= {V_abs_loo[0].cpu().numpy()}")
    print(f"[DIAG] V_abs (self)= {V_abs_self[0].cpu().numpy()}")
    print(f"[DIAG] V_sgn (loo )= {V_sgn_loo[0].cpu().numpy()}")
    print(f"[DIAG] V_sgn (self)= {V_sgn_self[0].cpu().numpy()}")
    print(f"[DIAG] ||Δ_abs||={float(torch.norm(V_abs_self - V_abs_loo)):.6f}  ||Δ_sgn||={float(torch.norm(V_sgn_self - V_sgn_loo)):.6f}")
    # purely diagnostic — no assert


@pytest.mark.e2e
def test_diag_portfolio_id_vs_manual_ctx_equivalence(tmp_path: Path, capfd):
    """
    Build a basket with portfolio_id, capture scorer preds,
    then rebuild port_ctx manually and run direct forward — print both and the max diff.
    Helps verify runtime ctx building equivalence.
    """
    work, pyg_dir, out, ctx = _generate_toy_world(tmp_path / "toy_pid_ctx", seed=83)
    cfg = MVDGTTrainConfig(
        workdir=work, pyg_dir=pyg_dir, outdir=out,
        epochs=15, lr=5e-3, weight_decay=1e-4, batch_size=128,
        seed=19, device="cpu", enable_tb=False, enable_tqdm=False,
        model=MVDGTModelConfig(hidden=64, heads=2, dropout=0.10, trade_dim=2, use_portfolio=True),
    )
    train_mvdgt(cfg)

    scorer = DGTScorer.from_dir(out)
    state = _rebuild_state_for_manual_forward(out)
    device=state["device"]; model=state["model"]; x=state["x"]

    asof = str(pd.Timestamp(ctx["market_dates"].iloc[-1].asof_date).date())
    rows = [
        {"portfolio_id":"PZ", "isin": _make_isin(7), "side":"BUY",  "size": 1.8e5, "asof_date":asof},
        {"portfolio_id":"PZ", "isin": _make_isin(8), "side":"SELL", "size": 1.0e5, "asof_date":asof},
        {"portfolio_id":"PZ", "isin": _make_isin(9), "side":"BUY",  "size": 2.5e5, "asof_date":asof},
    ]
    y_scorer = scorer.score_many(rows)

    anchor_idx, trade, market_feat = _vectorize_rows_for_state(state, rows)
    pf_gid = torch.zeros((len(rows),), dtype=torch.long, device=device)
    node_ids = anchor_idx.detach().cpu().tolist()
    port_ctx = _build_manual_runtime_ctx(rows, node_ids, device)

    with torch.no_grad():
        y_direct = model(x, anchor_idx=anchor_idx, market_feat=market_feat,
                         pf_gid=pf_gid, port_ctx=port_ctx, trade_feat=trade).cpu().numpy().astype(np.float32)
    print(f"[DIAG] scorer preds: {np.round(y_scorer,6).tolist()}")
    print(f"[DIAG] direct preds: {np.round(y_direct,6).tolist()}")
    print(f"[DIAG] max|scorer - direct|={float(np.max(np.abs(y_scorer - y_direct))):.6f}")
    # purely diagnostic

@pytest.mark.unit
def test_diag_port_vectors_components(capfd):
    """
    Simple unit probe of the legacy, signed weighted-sum (_portfolio_vectors) for math sanity,
    plus the samplewise LOO vectors to compare raw/normalized/tanh-ed components.
    """
    torch.manual_seed(0)
    # H: 4 nodes × 2 dim
    H = torch.tensor([[1.,0.],[0.,1.],[1.,1.],[2.,1.]], dtype=torch.float32)
    port_ctx = {
        'port_nodes_flat': torch.tensor([1,3], dtype=torch.long),
        'port_w_signed_flat': torch.tensor([0.25, 0.75], dtype=torch.float32),
        'port_len': torch.tensor([2], dtype=torch.long),
    }
    from ptliq.model.mv_dgt import MultiViewDGT, compute_samplewise_portfolio_vectors_loo

    masks = {k: torch.zeros(4, dtype=torch.bool) for k in ["struct","port","corr_global","corr_local"]}
    model = MultiViewDGT(x_dim=2, hidden=2, heads=1, dropout=0.0,
                         view_masks=masks, edge_index=torch.zeros(2,4, dtype=torch.long))

    # pf_gid = torch.tensor([0, -1], dtype=torch.long)
    #
    # --- legacy per-sample signed weighted sum (no norm, no LOO) via adapter if present
    # has_legacy = hasattr(model, "_portfolio_vectors")
    # if has_legacy:
    #     out_legacy = model._portfolio_vectors(H, pf_gid, port_ctx)
    #     print(f"[DIAG] legacy _portfolio_vectors(H): {out_legacy.index(0).detach().cpu().numpy()}")
    # else:
    #     print("[DIAG] legacy _portfolio_vectors not present on model (adapter missing).")

    # --- strict LOO, samplewise
    anchor_idx = torch.tensor([3], dtype=torch.long)  # node 3 in the (1,3) group → LOO must leave only node 1
    V_abs, V_sgn = compute_samplewise_portfolio_vectors_loo(H, anchor_idx, torch.tensor([0]), port_ctx, l2_normalize=False)
    print(f"[DIAG] LOO_abs_raw={V_abs[0].tolist()}  LOO_sgn_raw={V_sgn[0].tolist()}  (should equal H[1] * weights / sum(abs w) if only 1 left)")
    # purely diagnostic
