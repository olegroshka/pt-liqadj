# tests/test_mvdgt_diagnostic_e2e.py
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import pytest
import torch

from ptliq.model.mv_dgt import MultiViewDGT
from tests.test_mvdgt_e2e import _make_isin, _direct_predict, _generate_toy_world

# Skip if torch_geometric isn't available
_tg = pytest.importorskip("torch_geometric", reason="torch-geometric is required for MV-DGT e2e test")

from ptliq.training.mvdgt_loop import train_mvdgt, MVDGTTrainConfig, MVDGTModelConfig
from ptliq.service.scoring import DGTScorer


# =========================
# DIAGNOSTIC TESTS (non-failing, print classification)
# =========================

def _pick_group_with_neg_drag(ctx):
    """Pick (g, anchor_nid, co1, co2) so that mean(co1,co2) projects most negatively on U_g."""
    X = ctx["X"]; U = ctx["U"]; portfolios = ctx["portfolios"]
    best = None
    for g, members in enumerate(portfolios):
        if len(members) < 3:
            continue
        members = list(map(int, members))
        u = U[g]
        proj = np.dot(X[members], u)
        a_local = int(np.argmax(proj))
        anchor = int(members[a_local])
        order = np.argsort(proj).tolist()
        co_candidates = [int(members[i]) for i in order if int(members[i]) != anchor]
        if len(co_candidates) < 2:
            continue
        co1, co2 = co_candidates[0], co_candidates[1]
        neg_mean = float(proj[order[0]] + proj[order[1]]) / 2.0
        separation = float(proj[a_local] - neg_mean)
        if (best is None) or (separation > best["sep"]):
            best = {"g": g, "anchor": anchor, "co": (co1, co2), "sep": separation}
    return best  # dict or None

def _strip_pf(rows):
    """Remove portfolio_id/pf_gid in-place clone."""
    out = []
    for r in rows:
        d = dict(r)
        d.pop("portfolio_id", None)
        d.pop("pf_gid", None)
        out.append(d)
    return out

@pytest.fixture(scope="module")
def _diag_setup(tmp_path_factory):
    """Train once and reuse across diagnostics."""
    tmp = tmp_path_factory.mktemp("toy_diag_suite")
    work, pyg_dir, out, ctx = _generate_toy_world(tmp / "toy_diag", seed=70)
    cfg = MVDGTTrainConfig(
        workdir=work, pyg_dir=pyg_dir, outdir=out,
        epochs=30, lr=5e-3, weight_decay=1e-4, batch_size=128,
        seed=31, device="cpu", enable_tb=False, enable_tqdm=False,
        model=MVDGTModelConfig(hidden=64, heads=2, dropout=0.10, trade_dim=2, use_portfolio=True),
    )
    metrics = train_mvdgt(cfg)
    scorer = DGTScorer.from_dir(out)
    return {"work": work, "pyg_dir": pyg_dir, "out": out, "ctx": ctx, "scorer": scorer, "metrics": metrics}

@pytest.mark.e2e
def test_mvdgt_diag_runtime_vs_signed_behavior(_diag_setup):
    """
    DIAGNOSTIC: Compare anchor deltas for:
      - sign-cancelled co-items (BUY & SELL equal sizes)
      - same-sign co-items (both SELL)
      - strip portfolio context (static graph only)
    Prints 'DIAG | ' classification lines. Does not fail.
    """
    ctx = _diag_setup["ctx"]; scorer = _diag_setup["scorer"]; out = _diag_setup["out"]
    pick = _pick_group_with_neg_drag(ctx)
    assert pick is not None, "No group with >=3 members to build diagnostics"
    g = int(pick["g"]); anchor_nid = int(pick["anchor"]); co1, co2 = map(int, pick["co"])
    isin_anchor = _make_isin(anchor_nid); isin1 = _make_isin(co1); isin2 = _make_isin(co2)
    asof = str(pd.Timestamp(ctx["market_dates"].iloc[-1].asof_date).date())
    base_size = 1.5e5; co_size = 2.0e5

    # Anchor side taken from train row distribution: test both BUY and SELL
    sides = ["BUY", "SELL"]
    results = []
    for side_anchor in sides:
        # P0: anchor only
        P0 = [{"portfolio_id": "D0", "pf_gid": 0, "isin": isin_anchor, "side": side_anchor, "size": base_size, "asof_date": asof}]
        # P_cancel: add BUY&SELL co-items same sizes (signed sum ≈ 0)
        P_cancel = P0 + [
            {"portfolio_id": "D0", "pf_gid": 0, "isin": isin1, "side": "BUY",  "size": co_size, "asof_date": asof},
            {"portfolio_id": "D0", "pf_gid": 0, "isin": isin2, "side": "SELL", "size": co_size, "asof_date": asof},
        ]
        # P_same: both SELL
        P_same = P0 + [
            {"portfolio_id": "D1", "pf_gid": 1, "isin": isin1, "side": "SELL", "size": co_size, "asof_date": asof},
            {"portfolio_id": "D1", "pf_gid": 1, "isin": isin2, "side": "SELL", "size": co_size, "asof_date": asof},
        ]

        y0 = float(scorer.score_many(P0)[0])
        y_cancel = float(scorer.score_many(P_cancel)[0])
        y_same   = float(scorer.score_many(P_same)[0])

        # Strip portfolio context (runtime path off)
        y0_nopf = float(scorer.score_many(_strip_pf(P0))[0])
        y_cancel_nopf = float(scorer.score_many(_strip_pf(P_cancel))[0])

        results.append({
            "side": side_anchor,
            "Δ_cancel": y_cancel - y0,
            "Δ_same":   y_same   - y0,
            "Δ_collapse": (y_cancel - y0) - (y_cancel_nopf - y0_nopf),
        })

    # Gate snapshot if available
    pf_gate, port_gate, corr_gate = None, None, None
    try:
        m = scorer.model
        if hasattr(m, "pf_gate"):
            pf_gate = float(torch.sigmoid(getattr(m, "pf_gate")).detach().cpu().item())
        if hasattr(m, "portfolio_gate"):
            port_gate = float(torch.sigmoid(getattr(m, "portfolio_gate")).detach().cpu().item())
        if hasattr(m, "corr_gate"):
            corr_gate = float(torch.sigmoid(getattr(m, "corr_gate")).detach().cpu().item())
    except Exception:
        pass

    print("\nDIAG | ===== Runtime vs Signed Behavior =====")
    print(f"DIAG | group={g}, anchor={anchor_nid}, co={co1, co2}, β_pf={ctx.get('beta_pf', 'NA')}")
    print(f"DIAG | gates: pf_gate={pf_gate}, portfolio_gate={port_gate}, corr_gate={corr_gate}")
    for r in results:
        print(f"DIAG | side={r['side']:<4}  Δ_cancel={r['Δ_cancel']:.3f} bps   Δ_same={r['Δ_same']:.3f} bps   collapse(no-pf)={r['Δ_collapse']:.3f} bps")

    # Lightweight classification (printed, not asserted)
    Δc = np.array([r["Δ_cancel"] for r in results], dtype=np.float32)
    Δs = np.array([r["Δ_same"]   for r in results], dtype=np.float32)
    coll = np.array([r["Δ_collapse"] for r in results], dtype=np.float32)

    def _med(a): return float(np.median(a)) if a.size else float("nan")
    print("DIAG | summary: median(Δ_cancel)={:.3f}  median(Δ_same)={:.3f}  median(collapse)={:.3f}".format(_med(Δc), _med(Δs), _med(coll)))

    if np.all(np.abs(coll) < 0.03):
        print("DIAG | classification: runtime portfolio path appears INERT; effect likely from static graph views.")
    elif np.median(np.abs(Δs)) > 3.0 * max(1e-6, np.median(np.abs(Δc))):
        print("DIAG | classification: model is SIGN-SENSITIVE (uses signed weights); generator expects sign-AGNOSTIC abs pooling.")
    elif np.all(Δc > 0.0):
        print("DIAG | classification: adding co-items lifts anchor for both sides -> likely ANCHOR-INCLUSIVE pooling.")
    else:
        print("DIAG | classification: mixed; check manual ctx probe below.")

    # test is informational; do not fail
    assert True

# ---------- Manual port_ctx probe (abs vs signed; LOO vs self-inclusion) ----------

def _rebuild_state_for_manual_forward(outdir: Path) -> Dict[str, Any]:
    """Minimal reimplementation of _direct_predict() that returns a state dict
    and allows passing a manual port_ctx into model.forward."""
    device = torch.device("cpu")
    ckpt = torch.load(outdir / "ckpt.pt", map_location="cpu")
    model_cfg = MVDGTModelConfig(**ckpt["model_config"])
    meta = json.loads((outdir / "mvdgt_meta.json").read_text())
    view_masks = torch.load(outdir / "view_masks.pt", map_location="cpu")

    # graph + contexts
    data = torch.load(Path(meta["files"]["pyg_graph"]), map_location="cpu", weights_only=False)
    x = data.x.float().to(device)
    edge_index = data.edge_index.to(device)
    edge_weight = data.edge_weight.to(device) if hasattr(data, "edge_weight") else None
    vm = {k: v.to(device) for k, v in view_masks.items()}

    # market
    mkt_ctx = None; mkt_lookup = None
    pre_mean, pre_std, pre_sign = None, None, 1.0
    mkt_ctx_path = meta["files"].get("market_context")
    mkt_idx_path = meta["files"].get("market_index")
    if mkt_ctx_path and Path(mkt_ctx_path).exists():
        mkt_ctx = torch.load(mkt_ctx_path, map_location=device)
        if mkt_idx_path and Path(mkt_idx_path).exists():
            idx_df = pd.read_parquet(mkt_idx_path)
            idx_df["asof_date"] = pd.to_datetime(idx_df["asof_date"]).dt.normalize()
            mkt_lookup = {pd.Timestamp(r.asof_date): int(r.row_idx) for r in idx_df.itertuples(index=False)}
    preproc_path = outdir / "market_preproc.json"
    if preproc_path.exists():
        pre = json.loads(preproc_path.read_text())
        pre_mean = torch.tensor(pre.get("mean", []), dtype=torch.float32, device=device)
        std_vals = [float(v) if (v is not None and float(v) > 0.0) else 1.0 for v in pre.get("std", [])]
        pre_std = torch.tensor(std_vals, dtype=torch.float32, device=device)
        pre_sign = float(pre.get("sign", 1.0))

    # nodes
    nodes = pd.read_parquet(meta["files"]["graph_nodes"])
    isin_to_node = {str(r.isin): int(r.node_id) for r in nodes.itertuples(index=False)}

    # scaler + feature order
    fnames = json.loads((outdir / "feature_names.json").read_text())
    sc = json.loads((outdir / "scaler.json").read_text())
    mean = torch.tensor([float(sc["mean"][i]) for i in range(len(fnames))], dtype=torch.float32, device=device)
    std  = torch.tensor([float(sc["std"][i])  if (float(sc["std"][i]) > 0.0) else 1.0
                         for i in range(len(fnames))], dtype=torch.float32, device=device)

    # model
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
    ).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    return {
        "device": device, "model": model, "x": x,
        "fnames": fnames, "mean": mean, "std": std,
        "mkt_ctx": mkt_ctx, "mkt_lookup": mkt_lookup,
        "mkt_mean": pre_mean, "mkt_std": pre_std, "mkt_sign": pre_sign,
        "isin_to_node": isin_to_node,
    }

def _vectorize_rows_for_state(state: Dict[str, Any], rows: list[dict]):
    """Build anchor_idx, trade_feat, market_feat tensors to feed model.forward."""
    device = state["device"]; fnames = state["fnames"]; mean = state["mean"]; std = state["std"]
    isin_to_node = state["isin_to_node"]; mkt_ctx = state["mkt_ctx"]; mkt_lookup = state["mkt_lookup"]
    mkt_mean, mkt_std, mkt_sign = state["mkt_mean"], state["mkt_std"], state["mkt_sign"]

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

    # nodes
    node_ids = []
    for r in rows:
        isin = str(r.get("isin","")).strip()
        if isin not in isin_to_node:
            raise ValueError(f"Unknown ISIN: {isin}")
        node_ids.append(isin_to_node[isin])
    anchor_idx = torch.as_tensor(node_ids, dtype=torch.long, device=device)

    # trade features
    side = [_side_sign(r.get("side")) for r in rows]
    lsz  = [_log_size(r.get("size")) for r in rows]
    n = len(rows)
    feat_cols = []
    for i, name in enumerate(fnames):
        if name == "side_sign":
            feat_cols.append(torch.as_tensor(side, dtype=torch.float32, device=device))
        elif name == "log_size":
            feat_cols.append(torch.as_tensor(lsz, dtype=torch.float32, device=device))
        else:
            feat_cols.append(torch.full((n,), float(mean[i].item()), dtype=torch.float32, device=device))
    raw = torch.stack(feat_cols, dim=1) if feat_cols else torch.zeros((n,0), dtype=torch.float32, device=device)
    trade = torch.nan_to_num((raw - mean) / torch.where(std <= 0, torch.ones_like(std), std),
                             nan=0.0, posinf=0.0, neginf=0.0)

    # market features
    market_feat = None
    if mkt_ctx is not None:
        idxs = []
        last_idx = int(mkt_ctx["mkt_feat"].size(0) - 1)
        if mkt_lookup:
            items = sorted(mkt_lookup.items())
            dates_index = pd.to_datetime([k for k, _ in items])
            idx_arr = np.asarray([v for _, v in items], dtype=np.int64)
        else:
            dates_index, idx_arr = None, None
        for r in rows:
            ts = _to_date(r.get("asof_date"))
            if (ts is None) or (mkt_lookup is None):
                idxs.append(last_idx)
                continue
            if ts in mkt_lookup:
                idxs.append(int(mkt_lookup[ts]))
            elif (dates_index is None) or (len(idx_arr) == 0):
                idxs.append(last_idx)
            else:
                pos = int(dates_index.searchsorted(ts, side="right") - 1)
                pos = max(0, min(pos, len(idx_arr) - 1))
                idxs.append(int(idx_arr[pos]))
        di = torch.as_tensor(idxs, dtype=torch.long, device=device)
        market_feat = mkt_ctx["mkt_feat"].index_select(0, di)
        if (mkt_mean is not None) and (mkt_std is not None):
            denom = torch.where(mkt_std <= 0, torch.ones_like(mkt_std), mkt_std)
            market_feat = (market_feat - mkt_mean) / denom
            market_feat = torch.nan_to_num(market_feat, nan=0.0, posinf=0.0, neginf=0.0)
            if mkt_sign is not None:
                market_feat = market_feat * float(mkt_sign)

    return anchor_idx, trade, market_feat

def _manual_port_ctx(device, group_nodes: list[int], w_abs: list[float], w_signed: list[float]):
    return {
        "port_nodes_flat": torch.tensor(group_nodes, dtype=torch.long, device=device),
        "port_w_abs_flat": torch.tensor(w_abs, dtype=torch.float32, device=device),
        "port_w_signed_flat": torch.tensor(w_signed, dtype=torch.float32, device=device),
        "port_len": torch.tensor([len(group_nodes)], dtype=torch.long, device=device),
    }

@pytest.mark.e2e
def test_mvdgt_diag_manual_port_ctx_probe(_diag_setup):
    """
    DIAGNOSTIC: Bypass scorer and inject manual port_ctx to test:
      - abs-only vs signed baskets (same abs weights)
      - leave-one-out vs self-inclusive behavior
    Prints 'DIAG | ' block, does not fail.
    """
    out = _diag_setup["out"]; ctx = _diag_setup["ctx"]
    state = _rebuild_state_for_manual_forward(out)
    device = state["device"]; model = state["model"]; x = state["x"]

    pick = _pick_group_with_neg_drag(ctx)
    assert pick is not None, "No group with >=3 members to build diagnostics"
    g = int(pick["g"]); anchor_nid = int(pick["anchor"]); co1, co2 = map(int, pick["co"])
    isin_anchor = _make_isin(anchor_nid); asof = str(pd.Timestamp(ctx["market_dates"].iloc[-1].asof_date).date())

    # Single-anchor row (BUY; SELL behaves similarly for these probes)
    base_size = 1.5e5
    rows = [{"isin": isin_anchor, "side": "BUY", "size": base_size, "asof_date": asof}]
    anchor_idx, trade, market_feat = _vectorize_rows_for_state(state, rows)

    # pf_gid: one anchor, single group id 0
    pf_gid = torch.zeros((1,), dtype=torch.long, device=device)

    # --- Abs-only LOO: co1 & co2 only, equal abs weights; signed zero
    ctx_abs = _manual_port_ctx(device, [co1, co2], [0.5, 0.5], [0.0, 0.0])
    # --- Signed variant: same abs, but opposing signs (+0.5, -0.5)
    ctx_signed = _manual_port_ctx(device, [co1, co2], [0.5, 0.5], [0.5, -0.5])
    # --- Self-inclusive: include anchor as a third member with some abs weight
    ctx_self_inc = _manual_port_ctx(device, [anchor_nid, co1, co2], [0.34, 0.33, 0.33], [0.34, 0.33, -0.33])

    with torch.no_grad():
        y0 = float(model(x, anchor_idx=anchor_idx, market_feat=market_feat, pf_gid=pf_gid, port_ctx=None, trade_feat=trade).cpu().numpy().reshape(-1)[0])
        y_abs = float(model(x, anchor_idx=anchor_idx, market_feat=market_feat, pf_gid=pf_gid, port_ctx=ctx_abs, trade_feat=trade).cpu().numpy().reshape(-1)[0])
        y_sgn = float(model(x, anchor_idx=anchor_idx, market_feat=market_feat, pf_gid=pf_gid, port_ctx=ctx_signed, trade_feat=trade).cpu().numpy().reshape(-1)[0])
        y_self = float(model(x, anchor_idx=anchor_idx, market_feat=market_feat, pf_gid=pf_gid, port_ctx=ctx_self_inc, trade_feat=trade).cpu().numpy().reshape(-1)[0])

    Δ_abs = y_abs - y0
    Δ_sgn = y_sgn - y0
    Δ_self = y_self - y0

    # Gate snapshot if available
    pf_gate, port_gate, corr_gate = None, None, None
    try:
        if hasattr(model, "pf_gate"):
            pf_gate = float(torch.sigmoid(getattr(model, "pf_gate")).detach().cpu().item())
        if hasattr(model, "portfolio_gate"):
            port_gate = float(torch.sigmoid(getattr(model, "portfolio_gate")).detach().cpu().item())
        if hasattr(model, "corr_gate"):
            corr_gate = float(torch.sigmoid(getattr(model, "corr_gate")).detach().cpu().item())
    except Exception:
        pass

    print("\nDIAG | ===== Manual port_ctx probe =====")
    print(f"DIAG | group={g}, anchor={anchor_nid}, co={co1, co2}")
    print(f"DIAG | gates(model): pf_gate={pf_gate}, portfolio_gate={port_gate}, corr_gate={corr_gate}")
    print(f"DIAG | y0={y0:.3f}  y_abs(LOO)={y_abs:.3f}  y_signed={y_sgn:.3f}  y_selfinc={y_self:.3f}")
    print(f"DIAG | Δ_abs(LOO)={Δ_abs:.3f}  Δ_signed={Δ_sgn:.3f}  Δ_selfinc={Δ_self:.3f}")

    # Heuristic classification (printed)
    if abs(Δ_abs) < 0.03 and abs(Δ_sgn) >= 3 * max(0.01, abs(Δ_abs)):
        print("DIAG | classification: SIGN path dominates; ABS-LOO likely ignored.")
    elif Δ_self > max(Δ_abs, Δ_sgn) + 0.05:
        print("DIAG | classification: SELF-INCLUSIVE pooling likely; anchor contributes to its own portfolio vector.")
    elif abs(Δ_abs) < 0.03 and abs(Δ_sgn) < 0.03:
        print("DIAG | classification: Portfolio path effect is NUMERICALLY NEGLIGIBLE (gate ~ closed or head unused).")
    else:
        print("DIAG | classification: Mixed; ABS-LOO has some effect.")

    assert True

@pytest.mark.e2e
def test_mvdgt_diag_masks_and_context_snapshot(_diag_setup):
    """
    DIAGNOSTIC: print view-mask sizes and confirm scorer vs direct parity on a trivial basket.
    """
    scorer = _diag_setup["scorer"]; out = _diag_setup["out"]; ctx = _diag_setup["ctx"]
    # tiny parity check on 1 row
    row = [{"isin": _make_isin(0), "side": "BUY", "size": 1e5, "asof_date": str(pd.Timestamp(ctx["market_dates"].iloc[-1].asof_date).date())}]
    p_scorer = scorer.score_many(row)[0]
    p_direct = _direct_predict(out, row)[0]
    print("\nDIAG | ===== View masks & parity =====")
    print(f"DIAG | scorer vs direct parity (1 row): |Δ|={abs(float(p_scorer - p_direct)):.6f} bps")

    # print mask sizes via model attributes if exposed
    try:
        m = scorer.model
        counts = {}
        for name in ["struct","port","corr_global","corr_local"]:
            t = getattr(m, f"mask_{name}", None)
            if t is not None:
                counts[name] = int(t.sum().item())
        print(f"DIAG | view mask counts: {counts}")
    except Exception:
        print("DIAG | view mask counts: <unavailable>")

    assert True

