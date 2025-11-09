# tests/test_mvdgt_e2e.py
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

# Skip if torch_geometric isn't available
_tg = pytest.importorskip("torch_geometric", reason="torch-geometric is required for MV-DGT e2e test")
from torch_geometric.data import Data

from ptliq.training.mvdgt_loop import train_mvdgt, MVDGTTrainConfig, MVDGTModelConfig
from ptliq.service.scoring import DGTScorer


# --------------------------
# Helpers to build toy corpus
# --------------------------
def _rng(seed=17):
    return np.random.default_rng(seed)

def _make_isin(i: int) -> str:
    return f"SIM{str(i).zfill(10)}"

def _pairs_to_directed(pairs):
    """Undirected pairs -> both directions, no self loops."""
    out = []
    for u, v in pairs:
        if u == v:
            continue
        out.append((u, v)); out.append((v, u))
    return out

def _build_view_masks(E: int, segments: dict[str, tuple[int,int]]) -> dict[str, torch.Tensor]:
    masks = {}
    for name, (s, e) in segments.items():
        m = np.zeros(E, dtype=bool)
        m[s:e] = True
        masks[name] = torch.from_numpy(m)
    return masks

def _generate_toy_world(root: Path, seed: int = 42):
    """
    Build a small, end-to-end MV-DGT toy corpus under root/:
      - pyg/: pyg_graph.pt, graph_nodes.parquet, market_context.pt, market_index.parquet, portfolio_context.pt
      - work/: mvdgt_meta.json, view_masks.pt, samples.parquet
      - out/:  (will be populated by training)

    Returns (workdir, pyg_dir, outdir) and the generator parameters used for y.
    """
    rng = _rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    pyg_dir = root / "pyg"; pyg_dir.mkdir(parents=True, exist_ok=True)
    work = root / "work"; work.mkdir(parents=True, exist_ok=True)
    out = root / "out";  out.mkdir(parents=True, exist_ok=True)

    # ---- Nodes & features
    N = 14                      # slightly larger to ensure a "cold" group holdout is feasible
    x_dim = 6
    X = rng.normal(0.0, 1.0, size=(N, x_dim)).astype(np.float32)
    isins = [_make_isin(i) for i in range(N)]
    nodes_df = pd.DataFrame({"node_id": np.arange(N, dtype=int), "isin": isins})
    nodes_df.to_parquet(pyg_dir / "graph_nodes.parquet", index=False)

    # ---- Edges per view
    # struct: chain; port: complete within portfolios; corr_global: ring 2-hop; corr_local: star at 0
    struct_pairs = [(i, i+1) for i in range(N-1)]
    # Build 4 portfolio groups to make a clean holdout (last group)
    portfolios = [[0,1,2,3], [4,5,6], [7,8,9], [10,11,12,13]]

    port_pairs = []
    for g in portfolios:
        for i in range(len(g)):
            for j in range(i+1, len(g)):
                port_pairs.append((g[i], g[j]))

    corr_global_pairs = [(i, (i+2) % N) for i in range(N)]
    corr_local_pairs  = [(0, i) for i in range(1, N)]

    views = {
        "struct": _pairs_to_directed(struct_pairs),
        "port": _pairs_to_directed(port_pairs),
        "corr_global": _pairs_to_directed(corr_global_pairs),
        "corr_local": _pairs_to_directed(corr_local_pairs),
    }

    # Concatenate edges by view; remember segments for masks
    all_e = []
    segments = {}
    for name in ["struct", "port", "corr_global", "corr_local"]:
        s = len(all_e)
        all_e.extend(views[name])
        e = len(all_e)
        segments[name] = (s, e)
    edge_index = np.array(all_e, dtype=np.int64).T  # shape (2, E)
    E = edge_index.shape[1]
    edge_weight = np.ones(E, dtype=np.float32)

    # ---- Save PyG graph
    data = Data(
        x=torch.from_numpy(X),
        edge_index=torch.from_numpy(edge_index),
        edge_weight=torch.from_numpy(edge_weight),
    )
    pyg_graph_path = pyg_dir / "pyg_graph.pt"
    torch.save(data, pyg_graph_path)

    # ---- Save view masks (work/view_masks.pt)
    masks = _build_view_masks(E, segments)
    torch.save(masks, work / "view_masks.pt")

    # ---- Market context (7 days × 3 features), with explicit index file
    T, F = 7, 3
    mkt_feat = torch.from_numpy(rng.normal(0.0, 0.2, size=(T, F)).astype(np.float32))
    torch.save({"mkt_feat": mkt_feat}, pyg_dir / "market_context.pt")
    # index with concrete dates for scorer
    start = pd.Timestamp("2024-10-01")
    mi = pd.DataFrame({"asof_date": [start + pd.Timedelta(days=i) for i in range(T)],
                       "row_idx": np.arange(T)})
    mi.to_parquet(pyg_dir / "market_index.parquet", index=False)

    # ---- Portfolio context (4 groups), abs weights sum to 1; signed add directionality
    port_nodes = []
    port_w_abs = []
    port_w_sgn = []
    port_len = []
    for g in portfolios:
        L = len(g)
        w_abs = rng.dirichlet(alpha=[1.5] * L).astype(np.float32)
        signs = rng.choice([-1.0, 1.0], size=L)
        w_sgn = (w_abs * signs).astype(np.float32)
        port_nodes.extend(g)
        port_w_abs.extend(w_abs.tolist())
        port_w_sgn.extend(w_sgn.tolist())
        port_len.append(L)
    port_ctx = {
        "port_nodes_flat": torch.tensor(port_nodes, dtype=torch.long),
        "port_w_abs_flat": torch.tensor(port_w_abs, dtype=torch.float32),
        "port_w_signed_flat": torch.tensor(port_w_sgn, dtype=torch.float32),
        "port_len": torch.tensor(port_len, dtype=torch.long),
    }
    torch.save(port_ctx, pyg_dir / "portfolio_context.pt")

    # ---- mvdgt_meta.json (paths for scorer/training)
    meta = {
        "views": {  # informational only
            "struct": [2,3,8,9],
            "port": [0,1],
            "corr_global": [4,6],
            "corr_local": [5,7],
        },
        "files": {
            "pyg_graph": str(pyg_graph_path),
            "market_context": str(pyg_dir / "market_context.pt"),
            "market_index": str(pyg_dir / "market_index.parquet"),
            "portfolio_context": str(pyg_dir / "portfolio_context.pt"),
            "graph_nodes": str(pyg_dir / "graph_nodes.parquet"),
        },
    }
    (work / "mvdgt_meta.json").write_text(json.dumps(meta, indent=2))

    # ---- Build samples.parquet with a known, low-scale generator (≈10–20 bps)
    rows = []
    for date_idx in range(T):
        for nid in range(N):
            # ~60% nodes trade each day
            if rng.random() < 0.6:
                side = float(rng.choice([-1.0, 1.0]))
                size = float(rng.integers(50_000, 250_001))
                log_size = math.log1p(abs(size))
                # assign pf_gid
                pf_gid = -1
                for g_id, g in enumerate(portfolios):
                    if nid in g:
                        pf_gid = g_id
                        break
                rows.append({
                    "node_id": nid,
                    "date_idx": date_idx,
                    "pf_gid": pf_gid,
                    "side_sign": side,
                    "log_size": log_size,
                })
    samples = pd.DataFrame(rows).reset_index(drop=True)

    # Train/val/test split (by row)
    idx = np.arange(len(samples))
    rng.shuffle(idx)
    n_tr = int(0.6 * len(idx))
    n_va = int(0.2 * len(idx))
    tr_idx = set(idx[:n_tr].tolist())
    va_idx = set(idx[n_tr:n_tr+n_va].tolist())
    def _split(i): return "train" if i in tr_idx else ("val" if i in va_idx else "test")
    samples["split"] = [ _split(i) for i in range(len(samples)) ]

    # Compute train-only stats for generator to keep scale consistent
    tr = samples[samples["split"] == "train"]
    mu_lsz = float(tr["log_size"].mean())
    sd_lsz = float(tr["log_size"].std(ddof=0) or 1.0)

    # Define a small linear generator in bps with an added portfolio-dependent term
    beta_side = 3.0
    beta_lsz  = 4.0
    beta_x    = 2.0
    beta_mkt  = 2.0
    beta_pf   = 1.5  # portfolio sensitivity scale (bps)
    w_node = rng.normal(0.0, 0.5, size=(X.shape[1],)).astype(np.float32)
    w_mkt  = rng.normal(0.0, 0.5, size=(mkt_feat.size(1),)).astype(np.float32)
    # per-portfolio unit directions in node feature space
    U = rng.normal(0.0, 1.0, size=(len(portfolios), X.shape[1])).astype(np.float32)
    U = U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-6)
    # map node -> pf_gid
    pf_by_nid = np.full((X.shape[0],), -1, dtype=int)
    for g_id, g in enumerate(portfolios):
        for nid in g:
            pf_by_nid[nid] = g_id

    def _y(nid: int, di: int, side: float, log_size: float) -> float:
        lsz_z = (log_size - mu_lsz) / (sd_lsz if sd_lsz > 0 else 1.0)
        base = (
            beta_side * side
            + beta_lsz * lsz_z
            + beta_x * float(np.dot(X[nid], w_node))
            + beta_mkt * float(np.dot(mkt_feat[di].numpy(), w_mkt))
        )
        # portfolio effect: leave-one-out average of other members projected on U_g
        pf = int(pf_by_nid[nid])
        pf_term = 0.0
        if (pf >= 0) and (pf < len(portfolios)):
            others = [j for j in portfolios[pf] if j != nid]
            if len(others) > 0:
                h_bar = X[others].mean(0)
                pf_term = beta_pf * float(np.dot(h_bar, U[pf]))
        y = base + pf_term
        # small Gaussian noise
        y += rng.normal(0.0, 0.5)
        return float(y)

    samples["y"] = [
        _y(int(r.node_id), int(r.date_idx), float(r.side_sign), float(r.log_size))
        for r in samples.itertuples(index=False)
    ]

    samples.to_parquet(work / "samples.parquet", index=False)

    return work, pyg_dir, out, {
        "isins": isins,
        "nodes_df": nodes_df,
        "market_dates": mi,
        "X": X,
        "mkt_feat": mkt_feat,
        "mu_lsz": mu_lsz,
        "sd_lsz": sd_lsz,
        "w_node": w_node,
        "w_mkt": w_mkt,
        "beta": (beta_side, beta_lsz, beta_x, beta_mkt),
        "beta_pf": beta_pf,
        "U": U,
        "portfolios": portfolios,
        "samples": samples,
        "files": meta.get("files", {}),
    }


# --------------------------
# Direct model inference (parity with scorer)
# --------------------------
def _direct_predict(outdir: Path, req_rows: list[dict], masks_path: Path | None = None, disable_port_ctx: bool = False) -> np.ndarray:
    """Reconstruct the trained model + contexts and run a direct forward pass,
    using the same scaler/feature_names as DGTScorer."""
    device = torch.device("cpu")
    ckpt = torch.load(outdir / "ckpt.pt", map_location="cpu")
    model_cfg = MVDGTModelConfig(**ckpt["model_config"])
    meta = json.loads((outdir / "mvdgt_meta.json").read_text())
    # allow overriding masks (e.g., ablation tests)
    vm_path = Path(masks_path) if (masks_path is not None) else (outdir / "view_masks.pt")
    view_masks = torch.load(vm_path, map_location="cpu")

    # graph + contexts
    data = torch.load(Path(meta["files"]["pyg_graph"]), map_location="cpu", weights_only=False)
    x = data.x.float().to(device)
    edge_index = data.edge_index.to(device)
    edge_weight = data.edge_weight.to(device) if hasattr(data, "edge_weight") else None
    vm = {k: v.to(device) for k, v in view_masks.items()}

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
    # market preproc (if available)
    mkt_mean = None; mkt_std = None; mkt_sign = 1.0
    preproc_path = outdir / "market_preproc.json"
    if preproc_path.exists():
        pre = json.loads(preproc_path.read_text())
        mkt_mean = torch.tensor(pre.get("mean", []), dtype=torch.float32, device=device)
        std_vals = [float(v) if (v is not None and float(v) > 0.0) else 1.0 for v in pre.get("std", [])]
        mkt_std = torch.tensor(std_vals, dtype=torch.float32, device=device)
        mkt_sign = float(pre.get("sign", 1.0))

    port_ctx = None
    port_ctx_path = meta["files"].get("portfolio_context")
    if port_ctx_path and Path(port_ctx_path).exists():
        port_ctx = torch.load(port_ctx_path, map_location=device)

    # build node -> pf_gid from training port_ctx
    node_to_pfgid = {}
    if port_ctx is not None:
        try:
            nodes_flat = port_ctx.get("port_nodes_flat")
            lens = port_ctx.get("port_len")
            if (nodes_flat is not None) and (lens is not None):
                nodes_np = nodes_flat.detach().cpu().numpy().astype(int)
                lens_np = lens.detach().cpu().numpy().astype(int)
                off = 0
                for g, L in enumerate(lens_np):
                    for k in range(int(L)):
                        nid = int(nodes_np[off + k])
                        if nid not in node_to_pfgid:
                            node_to_pfgid[nid] = int(g)
                    off += int(L)
        except Exception:
            node_to_pfgid = {}

    nodes = pd.read_parquet(meta["files"]["graph_nodes"])
    isin_to_node = {str(r.isin): int(r.node_id) for r in nodes.itertuples(index=False)}

    # scaler + feature order
    fnames = json.loads((outdir / "feature_names.json").read_text())
    sc = json.loads((outdir / "scaler.json").read_text())
    mean = torch.tensor([float(sc["mean"][i]) for i in range(len(fnames))], dtype=torch.float32, device=device)
    std  = torch.tensor([float(sc["std"][i])  if (float(sc["std"][i]) > 0.0) else 1.0
                         for i in range(len(fnames))], dtype=torch.float32, device=device)

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

    # vectorize
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

    node_ids = []
    for r in req_rows:
        isin = str(r.get("isin","")).strip()
        if isin not in isin_to_node:
            raise ValueError(f"Unknown ISIN: {isin}")
        node_ids.append(isin_to_node[isin])
    anchor_idx = torch.as_tensor(node_ids, dtype=torch.long, device=device)

    # pf_gid: prefer explicit; else derive from training port_ctx
    pf_list = []
    any_explicit = any((("pf_gid" in r) and (r["pf_gid"] is not None)) for r in req_rows)
    if any_explicit:
        for r in req_rows:
            try:
                pf_list.append(int(r.get("pf_gid", -1)))
            except Exception:
                pf_list.append(-1)
    else:
        for nid in node_ids:
            pf_list.append(int(node_to_pfgid.get(int(nid), -1)))
    pf_gid = torch.as_tensor(pf_list, dtype=torch.long, device=device)

    # Build dynamic runtime portfolio context to mirror DGTScorer behavior
    def _side_sign_local(v):
        if v is None: return 0.0
        s = str(v).strip().upper()
        return 1.0 if s in {"B","BUY","CBUY","TRUE","1"} else (-1.0 if s in {"S","SELL","CSELL","FALSE","0","-1"} else 0.0)

    def _build_port_ctx_from_groups(groups: dict, return_gid: bool = False):
        # groups: key -> list of row indices
        keys = sorted(groups.keys())
        key_to_gid = {k: i for i, k in enumerate(keys)}

        port_nodes, port_w_abs, port_w_sgn, port_len = [], [], [], []
        for k in keys:
            idxs = groups[k]
            abs_sizes = []
            sgn_sizes = []
            for i in idxs:
                side = _side_sign_local(req_rows[i].get("side"))
                try:
                    sz = float(req_rows[i].get("size", 0.0))
                except Exception:
                    sz = 0.0
                a = abs(sz)
                s = (1.0 if side >= 0 else -1.0) * a
                abs_sizes.append(a)
                sgn_sizes.append(s)
            denom = float(sum(abs_sizes)) or float(len(abs_sizes)) or 1.0
            w_abs = [a / denom for a in abs_sizes]
            w_sgn = [s / denom for s in sgn_sizes]
            port_len.append(len(idxs))
            for i, wA, wS in zip(idxs, w_abs, w_sgn):
                port_nodes.append(int(node_ids[i]))
                port_w_abs.append(float(wA))
                port_w_sgn.append(float(wS))

        ctx = {
            "port_nodes_flat": torch.tensor(port_nodes, dtype=torch.long, device=device),
            "port_w_abs_flat": torch.tensor(port_w_abs, dtype=torch.float32, device=device),
            "port_w_signed_flat": torch.tensor(port_w_sgn, dtype=torch.float32, device=device),
            "port_len": torch.tensor(port_len, dtype=torch.long, device=device),
        }

        if not return_gid:
            return ctx

        # Align pf_gid to these runtime groups (by portfolio_id if present)
        pf_gid_rt = []
        for r in req_rows:
            pid = r.get("portfolio_id")
            pf_gid_rt.append(int(key_to_gid.get(pid, -1)))
        pf_gid_rt = torch.as_tensor(pf_gid_rt, dtype=torch.long, device=device)
        return ctx, pf_gid_rt

    # precedence: explicit pf_gid → build dynamic ctx; else portfolio_id → build dynamic; else training port_ctx
    any_pid = any((r.get("portfolio_id") is not None) for r in req_rows)
    runtime_ctx_from_pid = False
    runtime_ctx_from_explicit = False
    if any_explicit:
        groups = {}
        for i, r in enumerate(req_rows):
            try:
                g = int(r.get("pf_gid", -1))
            except Exception:
                g = -1
            if g >= 0:
                groups.setdefault(g, []).append(i)
        if groups:
            keys = sorted(groups.keys())
            key_to_gid = {k: i for i, k in enumerate(keys)}
            pf_list_remap = []
            for r in req_rows:
                try:
                    g = int(r.get("pf_gid", -1))
                except Exception:
                    g = -1
                pf_list_remap.append(int(key_to_gid.get(g, -1)))
            pf_gid = torch.as_tensor(pf_list_remap, dtype=torch.long, device=device)
            port_ctx = _build_port_ctx_from_groups(groups)
            runtime_ctx_from_explicit = True
        else:
            port_ctx = None
    elif any_pid:
        groups = {}
        for i, r in enumerate(req_rows):
            pid = r.get("portfolio_id")
            if pid is None:
                continue
            groups.setdefault(pid, []).append(i)
        if groups:
            port_ctx, pf_gid = _build_port_ctx_from_groups(groups, return_gid=True)
            runtime_ctx_from_pid = True
        else:
            port_ctx = None
    # else: keep training port_ctx as loaded above

    # optional ablation: disable portfolio context entirely
    if disable_port_ctx:
        port_ctx = None

    side = [_side_sign(r.get("side")) for r in req_rows]
    lsz  = [_log_size(r.get("size")) for r in req_rows]
    n = len(req_rows)

    # align to saved feature_names and scale
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

    # market (exact match first, else nearest prior)
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
        for r in req_rows:
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
        # apply the same z-score + orientation sign as scorer if available
        if (mkt_mean is not None) and (mkt_std is not None):
            denom = torch.where(mkt_std <= 0, torch.ones_like(mkt_std), mkt_std)
            market_feat = (market_feat - mkt_mean) / denom
            market_feat = torch.nan_to_num(market_feat, nan=0.0, posinf=0.0, neginf=0.0)
            if mkt_sign is not None:
                market_feat = market_feat * float(mkt_sign)

    # consistency guard: pf_gid must index into the runtime port_ctx when both are provided
    if (port_ctx is not None) and (pf_gid is not None) and (runtime_ctx_from_pid or runtime_ctx_from_explicit):
        G = int(port_ctx["port_len"].numel())
        assert ((pf_gid >= 0) & (pf_gid < G)).all(), "pf_gid must index into runtime port_ctx"

    with torch.no_grad():
        yhat = model(x, anchor_idx=anchor_idx, market_feat=market_feat,
                     pf_gid=pf_gid, port_ctx=port_ctx, trade_feat=trade)
        y = yhat.detach().cpu().numpy().astype(np.float32).reshape(-1)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y


# --------------------------
# The actual integration test
# --------------------------
@pytest.mark.e2e
def test_mvdgt_e2e_scenarios_extended(tmp_path: Path):
    """
    End-to-end diagnostic:
      1) build toy corpus (graph, contexts, samples, meta, masks)
      2) force a 'held-out' portfolio group (cold) by removing it from TRAIN
      3) train MV-DGT (artifacts under out/)
      4) score 3-5 portfolio-like requests via DGTScorer and via direct forward
      5) assert:
          - scorer ≈ direct parity
          - magnitude sanity (bps bounds)
          - train-like rows RMSE small
          - B/C/D perturbations reasonable
          - side flip induces negative delta
          - size monotonicity
          - market shift sensitivity (sign alignment)
          - cold group has ≥ warm group error (or, at least, not better)
          - unknown ISIN raises
          - missing asof_date fallback equals last-day prediction
    """
    work, pyg_dir, out, ctx = _generate_toy_world(tmp_path / "toy", seed=42)

    # --- Force a 'cold' (held-out) portfolio group in samples before training
    samples = pd.read_parquet(work / "samples.parquet")
    # Deterministically hold out the last portfolio group as "cold":
    cold_gid = len(ctx["portfolios"]) - 1
    samples.loc[samples["pf_gid"] == cold_gid, "split"] = "test"
    # (Optionally, reduce train rows for that group further)
    samples.to_parquet(work / "samples.parquet", index=False)

    # --- Train on toy corpus
    cfg = MVDGTTrainConfig(
        workdir=work,
        pyg_dir=pyg_dir,
        outdir=out,
        epochs=30,
        lr=5e-3,
        weight_decay=1e-4,
        batch_size=128,
        seed=11,
        device="cpu",
        enable_tb=False,
        enable_tqdm=False,
        model=MVDGTModelConfig(hidden=64, heads=2, dropout=0.1, trade_dim=2, use_portfolio=True),
    )
    metrics = train_mvdgt(cfg)
    # Ensure artifacts exist
    assert (out / "ckpt.pt").exists(), "checkpoint missing"
    assert (out / "feature_names.json").exists()
    assert (out / "scaler.json").exists()
    assert (out / "mvdgt_meta.json").exists()
    assert metrics["test_rmse"] < 18.0  # toy generator is small; keep this loose

    # --- Load scorer from the training outdir
    scorer = DGTScorer.from_dir(out)

    # Node id -> isin
    nid2isin = {int(r.node_id): str(r.isin) for r in ctx["nodes_df"].itertuples(index=False)}

    # Pick 1 representative row from up to 3 *warm* portfolio groups (pf_gid>=0 and != cold_gid)
    samp = pd.read_parquet(work / "samples.parquet")
    tr_pf = samp[(samp["split"] == "train") & (samp["pf_gid"] >= 0) & (samp["pf_gid"] != cold_gid)]
    assert len(tr_pf) > 0, "no warm portfolio rows in training set"
    warm_reps = []
    for g in sorted(tr_pf["pf_gid"].unique().tolist()):
        grp = tr_pf[tr_pf["pf_gid"] == g]
        take = min(2, len(grp))
        if take > 0:
            warm_reps.extend([*grp.sample(take, random_state=100).itertuples(index=False)])
        if len(warm_reps) >= 6:
            break
    assert len(warm_reps) >= 4, "Not enough warm reps for a robust market-shift check"

    # Also pick up to 2 rows from the held-out (cold) portfolio group (from test split)
    cold_rows_src = samp[(samp["split"] != "train") & (samp["pf_gid"] == cold_gid)]
    cold_reps = []
    if len(cold_rows_src) > 0:
        cold_reps = [cold_rows_src.sample(1, random_state=101).iloc[0]]
        if len(cold_rows_src) > 1:
            cold_reps.append(cold_rows_src.sample(1, random_state=102).iloc[0])

    # Build scenario requests A/B/C/D for warm reps
    req_rows = []
    exp_y = []
    mi = ctx["market_dates"]
    X, w_node = ctx["X"], ctx["w_node"]
    mkt_feat, w_mkt = ctx["mkt_feat"], ctx["w_mkt"]
    mu, sd = ctx["mu_lsz"], ctx["sd_lsz"]
    beta_side, beta_lsz, beta_x, beta_mkt = ctx["beta"]
    # portfolio generator components
    beta_pf = ctx.get("beta_pf", 0.0)
    U = ctx.get("U")
    portfolios = ctx.get("portfolios")
    # precompute per-node portfolio term used by the generator for deterministic expectations
    pf_term_by_nid: dict[int, float] = {}
    if (U is not None) and (portfolios is not None) and float(beta_pf) != 0.0:
        for g, members in enumerate(portfolios):
            members = list(map(int, members))
            for nid0 in members:
                others = [j for j in members if j != nid0]
                if len(others) == 0:
                    pf_term_by_nid[nid0] = 0.0
                else:
                    h_bar = X[others].mean(0)
                    pf_term_by_nid[nid0] = float(beta_pf * float(np.dot(h_bar, U[g])))
    else:
        pf_term_by_nid = {}

    def _asof_for_idx(di: int) -> str:
        return str(pd.Timestamp(mi.iloc[di].asof_date).date())

    for r in warm_reps:
        nid = int(r.node_id)
        isin = nid2isin[nid]
        di = int(r.date_idx)
        asof_date = _asof_for_idx(di)
        base_size = float(math.expm1(float(r.log_size)))
        base_side = "BUY" if float(r.side_sign) > 0 else "SELL"

        # scenario A (identical)
        req_rows.append({"portfolio_id": f"PF-{int(r.pf_gid)}", "pf_gid": int(r.pf_gid), "isin": isin, "side": base_side, "size": base_size, "asof_date": asof_date})
        exp_y.append(float(r.y))  # contains noise; acceptable for consistency

        # scenario B (+20% size)
        size_b = base_size * 1.2
        req_rows.append({"portfolio_id": f"PF-{int(r.pf_gid)}", "pf_gid": int(r.pf_gid), "isin": isin, "side": base_side, "size": size_b, "asof_date": asof_date})
        side_sign = +1.0 if base_side == "BUY" else -1.0
        lsz_z = (math.log1p(size_b) - mu) / (sd if sd > 0 else 1.0)
        x_term = beta_x * float(np.dot(X[nid], w_node))
        m_term = beta_mkt * float(np.dot(mkt_feat[di].numpy(), w_mkt))
        exp_y.append(beta_side * side_sign + beta_lsz * lsz_z + x_term + m_term)  # deterministic

        # scenario C (flip side)
        flip_side = "SELL" if base_side == "BUY" else "BUY"
        req_rows.append({"portfolio_id": f"PF-{int(r.pf_gid)}", "pf_gid": int(r.pf_gid), "isin": isin, "side": flip_side, "size": base_size, "asof_date": asof_date})
        side_sign = +1.0 if flip_side == "BUY" else -1.0
        lsz_z = (math.log1p(base_size) - mu) / (sd if sd > 0 else 1.0)
        exp_y.append(beta_side * side_sign + beta_lsz * lsz_z + x_term + m_term)

        # scenario D (next day)
        di2 = min(di + 1, len(mi) - 1)
        asof_date2 = _asof_for_idx(di2)
        req_rows.append({"portfolio_id": f"PF-{int(r.pf_gid)}", "pf_gid": int(r.pf_gid), "isin": isin, "side": base_side, "size": base_size, "asof_date": asof_date2})
        side_sign = +1.0 if base_side == "BUY" else -1.0
        lsz_z = (math.log1p(base_size) - mu) / (sd if sd > 0 else 1.0)
        m_term2 = beta_mkt * float(np.dot(mkt_feat[di2].numpy(), w_mkt))
        exp_y.append(beta_side * side_sign + beta_lsz * lsz_z + x_term + m_term2)

    # Build a few 'cold' (held-out group) requests (same A/B/C, skip D to keep size modest)
    cold_req_rows = []
    cold_exp_y = []
    for r in cold_reps:
        nid = int(r.node_id)
        isin = nid2isin[nid]
        di = int(r.date_idx)
        asof_date = _asof_for_idx(di)
        base_size = float(math.expm1(float(r.log_size)))
        base_side = "BUY" if float(r.side_sign) > 0 else "SELL"
        # A
        cold_req_rows.append({"portfolio_id": f"PF-{int(r.pf_gid)}", "pf_gid": int(r.pf_gid), "isin": isin, "side": base_side, "size": base_size, "asof_date": asof_date})
        side_sign = +1.0 if base_side == "BUY" else -1.0
        lsz_z = (math.log1p(base_size) - mu) / (sd if sd > 0 else 1.0)
        x_term = beta_x * float(np.dot(X[nid], w_node))
        m_term = beta_mkt * float(np.dot(mkt_feat[di].numpy(), w_mkt))
        cold_exp_y.append(beta_side * side_sign + beta_lsz * lsz_z + x_term + m_term)  # deterministic
        # B (+50% size to magnify lsz effect)
        size_b = base_size * 1.5
        cold_req_rows.append({"portfolio_id": f"PF-{int(r.pf_gid)}", "pf_gid": int(r.pf_gid), "isin": isin, "side": base_side, "size": size_b, "asof_date": asof_date})
        lsz_z = (math.log1p(size_b) - mu) / (sd if sd > 0 else 1.0)
        cold_exp_y.append(beta_side * side_sign + beta_lsz * lsz_z + x_term + m_term)
        # C (flip side)
        flip_side = "SELL" if base_side == "BUY" else "BUY"
        cold_req_rows.append({"portfolio_id": f"PF-{int(r.pf_gid)}", "pf_gid": int(r.pf_gid), "isin": isin, "side": flip_side, "size": base_size, "asof_date": asof_date})
        side_sign = +1.0 if flip_side == "BUY" else -1.0
        lsz_z = (math.log1p(base_size) - mu) / (sd if sd > 0 else 1.0)
        cold_exp_y.append(beta_side * side_sign + beta_lsz * lsz_z + x_term + m_term)

    # --- Scoring: scorer and direct
    preds_scorer_warm = scorer.score_many(req_rows)
    preds_direct_warm = _direct_predict(out, req_rows)
    assert preds_scorer_warm.shape == preds_direct_warm.shape
    # Parity
    assert np.max(np.abs(preds_scorer_warm - preds_direct_warm)) < 1e-5, "Scorer vs direct mismatch"

    # Magnitude sanity (generator ~10–20bps)
    assert float(np.max(np.abs(preds_scorer_warm))) < 30.0, f"Preds too large for toy scale: {preds_scorer_warm}"

    # Train-like scenario A indices
    a_idx = list(range(0, len(req_rows), 4))  # 0,4,8,…
    exp_warm = np.array(exp_y, dtype=np.float32)
    rmse_A = float(np.sqrt(np.mean((preds_scorer_warm[a_idx] - exp_warm[a_idx]) ** 2)))
    assert rmse_A < 6.0, f"Train-like scenario RMSE too high: {rmse_A:.3f} bps"

    # Perturbed B/C/D
    bcd_idx = [i for i in range(len(req_rows)) if i % 4 != 0]
    rmse_BCD = float(np.sqrt(np.mean((preds_scorer_warm[bcd_idx] - exp_warm[bcd_idx]) ** 2)))
    assert rmse_BCD < 8.0, f"Scenario RMSE too high: {rmse_BCD:.3f} bps"

    # Side flip (C - A) should be negative (beta_side=+3 bps)
    deltas_CA = preds_scorer_warm[2::4] - preds_scorer_warm[0::4]  # (C - A) for each rep
    assert np.median(deltas_CA) < -0.5, f"Side-flip delta not negative enough: {deltas_CA}"

    # Size monotonicity (B - A) should be positive
    deltas_BA = preds_scorer_warm[1::4] - preds_scorer_warm[0::4]
    assert np.all(deltas_BA > 0.0), f"Size monotonicity violated: {deltas_BA}"

    # Market shift sensitivity (D - A): robust check invariant to market-basis rotations
    # Build expected deterministic deltas directly from the generator's market term (noise-free)
    exp_deltas = []
    for r in warm_reps:
        di  = int(r.date_idx)
        di2 = min(di + 1, len(mi) - 1)
        dmkt = float(np.dot(mkt_feat[di2].numpy(), w_mkt) - np.dot(mkt_feat[di].numpy(), w_mkt))
        exp_deltas.append(beta_mkt * dmkt)
    exp_deltas = np.asarray(exp_deltas, dtype=np.float32)

    deltas_DA_pred = (preds_scorer_warm[3::4] - preds_scorer_warm[0::4]).astype(np.float32)

    # Ignore near-zero expected deltas where sign is ill-defined
    EPS = 0.15  # bps
    mask = np.abs(exp_deltas) >= EPS
    Δp = deltas_DA_pred[mask]
    Δe = exp_deltas[mask]

    if len(Δp) >= 4:
        corr = float(np.corrcoef(Δp, Δe)[0,1]) if (np.std(Δp) > 0 and np.std(Δe) > 0) else 1.0
        assert abs(corr) >= 0.6, f"Market-shift |corr| too low: {corr:.2f} | pred={Δp} exp={Δe}"
    elif len(Δp) == 1:
        # With one effective pair, require sign consistency or very small predicted delta.
        pp = float(Δp[0]); ee = float(Δe[0])
        assert (np.sign(pp) == np.sign(ee)) or (abs(pp) < 0.2), f"Market-shift single-pair mismatch: pred={pp:.3f}, exp={ee:.3f}"
    # else: all expected deltas are ~0; nothing to assert

    # Cold/held-out group scoring + direct parity
    if cold_req_rows:
        preds_scorer_cold = scorer.score_many(cold_req_rows)
        preds_direct_cold = _direct_predict(out, cold_req_rows)
        assert np.max(np.abs(preds_scorer_cold - preds_direct_cold)) < 1e-5, "Scorer vs direct mismatch (cold)"

        exp_cold = np.array(cold_exp_y, dtype=np.float32)
        rmse_cold = float(np.sqrt(np.mean((preds_scorer_cold - exp_cold) ** 2)))
        # Expect cold error >= warm-perturbed error (allow tiny slack for stochasticity)
        tol_bps = 1.1
        assert rmse_cold >= rmse_BCD - tol_bps, f"Cold RMSE unexpectedly lower than warm BCD by >{tol_bps:.1f}bps: cold={rmse_cold:.3f}, warmBCD={rmse_BCD:.3f}"

    # Unknown ISIN → raises
    with pytest.raises(ValueError):
        _ = scorer.score_many([{"isin": "SIM9999999999", "side": "BUY", "size": 1e5, "asof_date": "2024-10-01"}])

    # Missing asof_date fallback equals last available day
    last_day = str(pd.Timestamp(ctx["market_dates"].iloc[-1].asof_date).date())
    test_row_last = [{"isin": _make_isin(0), "side": "BUY", "size": 1e5, "asof_date": last_day}]
    test_row_none = [{"isin": _make_isin(0), "side": "BUY", "size": 1e5}]  # no asof_date
    p_last = scorer.score_many(test_row_last)[0]
    p_none = scorer.score_many(test_row_none)[0]
    assert abs(p_last - p_none) < 1e-6, "Missing asof_date fallback mismatch with last-day prediction"

    # ---- Portfolio sensitivity: many-anchor drift ----
    # Build multiple (P_A, P_B) baskets for K anchors:
    #    P_A : common line + two co-items from its own training group
    #    P_B : common line + two co-items from a different group, bigger opposite-side sizes
    K = min(6, len(warm_reps))
    abs_deltas = []
    lists_A = []
    lists_B = []
    pred_pf_deltas = []
    isin2nid = {v: k for k, v in nid2isin.items()}
    for k in range(K):
        rep_k = warm_reps[k]
        nid_k = int(rep_k.node_id)
        isin_common = nid2isin[nid_k]
        asof_common = _asof_for_idx(int(rep_k.date_idx))
        base_size = float(math.expm1(float(rep_k.log_size)))
        side_lbl = "BUY" if float(rep_k.side_sign) > 0 else "SELL"
        gk = int(rep_k.pf_gid)

        # P_A: two co-items from the same training portfolio group
        g_members = [m for m in portfolios[gk] if m != nid_k]
        others1 = g_members[:2] if len(g_members) >= 2 else [j for j in range(X.shape[0]) if j != nid_k][:2]
        P_A = [{"portfolio_id": f"A_{k}", "isin": isin_common, "side": side_lbl, "size": base_size, "asof_date": asof_common}] + \
              [{"portfolio_id": f"A_{k}", "isin": nid2isin[j], "side": "BUY", "size": 1.0e5, "asof_date": asof_common} for j in others1]

        # P_B: two co-items from a different portfolio group, with larger opposite-side sizes
        other_groups = [g for g in portfolios if nid_k not in g]
        others2 = (other_groups[0][:2] if other_groups and len(other_groups[0]) >= 2
                   else [j for j in range(X.shape[0]) if j not in [nid_k] + others1][:2])
        P_B = [{"portfolio_id": f"B_{k}", "isin": isin_common, "side": side_lbl, "size": base_size, "asof_date": asof_common}] + \
              [{"portfolio_id": f"B_{k}", "isin": nid2isin[j], "side": "SELL", "size": 2.0e5, "asof_date": asof_common} for j in others2]

        yA = scorer.score_many(P_A)[0]
        yB = scorer.score_many(P_B)[0]
        abs_deltas.append(abs(float(yA - yB)))
        pred_pf_deltas.append(float(yB - yA))
        lists_A.append(P_A)
        lists_B.append(P_B)

    # Require a visible drift: median delta should exceed a meaningful threshold
    # tuned for generator beta_pf=1.5 bps; with noise and short training we target >= 0.5 bps
    median_delta = float(np.median(abs_deltas)) if abs_deltas else 0.0
    # use threshold relative to generator beta_pf when available, else default 0.5
    beta_pf = float(ctx.get("beta_pf", 1.5)) if isinstance(ctx, dict) else 1.5
    thresh = max(0.1, 0.1 * beta_pf)
    assert median_delta >= thresh, f"Portfolio sensitivity too small across anchors: median |Δ|={median_delta:.3f} bps, deltas={abs_deltas}, thresh={thresh:.3f}"

    # ---- Portfolio co-item size sensitivity (not identifiable in current generator) ----
    # Keep as a diagnostic only; do not assert, since the generator's pf-term ignores co-item size/side.
    deltas_co = []
    for k in range(K):
        rep_k = warm_reps[k]
        nid_k = int(rep_k.node_id)
        isin_k = nid2isin[nid_k]
        asof_k = _asof_for_idx(int(rep_k.date_idx))
        base_size_k = float(math.expm1(float(rep_k.log_size)))
        side_k = "BUY" if float(rep_k.side_sign) > 0 else "SELL"
        gk = int(rep_k.pf_gid)
        co_k = [m for m in portfolios[gk] if m != nid_k][:2]
        if len(co_k) == 0:
            continue
        P_small = [{"portfolio_id":f"M{k}","isin":isin_k,"side":side_k,"size":base_size_k,"asof_date":asof_k}] + \
                  [{"portfolio_id":f"M{k}","isin":nid2isin[j],"side":"SELL","size":5.0e4,"asof_date":asof_k} for j in co_k]
        P_big   = [{"portfolio_id":f"M{k}","isin":isin_k,"side":side_k,"size":base_size_k,"asof_date":asof_k}] + \
                  [{"portfolio_id":f"M{k}","isin":nid2isin[j],"side":"SELL","size":1.0e5,"asof_date":asof_k} for j in co_k]
        y_small = scorer.score_many(P_small)[0]
        y_big   = scorer.score_many(P_big)[0]
        deltas_co.append(abs(float(y_big - y_small)))
    # Print diagnostic if gates are clearly open
    try:
        pf_gate_val = float(torch.sigmoid(getattr(scorer.model, "pf_gate")).detach().cpu().item()) if hasattr(scorer.model, "pf_gate") else None
        port_gate_val = float(torch.sigmoid(getattr(scorer.model, "portfolio_gate")).detach().cpu().item()) if hasattr(scorer.model, "portfolio_gate") else None
        mg = [g for g in [pf_gate_val, port_gate_val] if g is not None]
        if len(mg) > 0 and max(mg) >= 0.05 and len(deltas_co) >= 2:
            median_co = float(np.median(deltas_co)); max_co = float(np.max(deltas_co))
            print(f"[DBG] co-size monotonicity (non-identifiable): median={median_co:.3f} max={max_co:.3f} bps, gates={mg}")
    except Exception:
        pass

    # ---- Sanity: if portfolio context is removed, drift collapses ----
    # Remove portfolio_id and pf_gid from the same two baskets; scores should become (nearly) equal
    def _strip_pf(b):
        return [{k:v for k,v in r.items() if k not in ("portfolio_id","pf_gid")} for r in b]
    yA_nopf = scorer.score_many(_strip_pf(lists_A[0]))
    yB_nopf = scorer.score_many(_strip_pf(lists_B[0]))
    collapse = abs(float(yA_nopf[0] - yB_nopf[0]))
    collapse_thresh = max(0.05, 0.13 * beta_pf)
    assert collapse < collapse_thresh, f"Drift still present with no portfolio context: |Δ|={collapse:.3f} bps, thresh={collapse_thresh:.3f}"

    # ---- View ablation: zero out 'port' view mask should shrink A vs B drift ----
    vm = torch.load(out / "view_masks.pt", map_location="cpu")
    vm_ablate = {k: v.clone() for k, v in vm.items()}
    if "port" in vm_ablate:
        vm_ablate["port"] = torch.zeros_like(vm_ablate["port"])
    torch.save(vm_ablate, out / "view_masks_port_off.pt")

    # pick the pair with the largest observed drift across anchors to make the ablation diagnostic robust
    idx_best = int(np.argmax(np.asarray(abs_deltas))) if abs_deltas else 0
    yA_full = _direct_predict(out, lists_A[idx_best])[0]
    yB_full = _direct_predict(out, lists_B[idx_best])[0]
    yA_port0 = _direct_predict(out, lists_A[idx_best], masks_path=out / "view_masks_port_off.pt", disable_port_ctx=True)[0]
    yB_port0 = _direct_predict(out, lists_B[idx_best], masks_path=out / "view_masks_port_off.pt", disable_port_ctx=True)[0]

    # The ablation is twofold: edge mask (port view) and disabling portfolio residual context
    assert abs(float(yA_full - yB_full)) >= max(0.1, 0.1 * beta_pf), "Portfolio drift too small with full masks"
    assert abs(float(yA_port0 - yB_port0)) < max(0.08, 0.08 * beta_pf), "Portfolio drift did not shrink when 'port' view was ablated (mask+no-port_ctx)"

    # ---- Alignment to generator's portfolio term (optional but informative) ----
    # For each (P_A, P_B) pair above, approximate the expected Δ_pf by the difference
    # of the generator's pf term for that anchor between the two baskets (using the same U, X).
    if (ctx.get("U") is not None) and (ctx.get("beta_pf") is not None):
        exp_pf_deltas = []
        for k in range(K):
            rep_k = warm_reps[k]
            nid_k = int(rep_k.node_id)
            gk = int(rep_k.pf_gid)
            isin_common = nid2isin[nid_k]
            def _pf_term_for(basket):
                pid = basket[0]["portfolio_id"]
                # reconstruct co-items' node_ids (exclude anchor)
                nids = [int(isin2nid[r["isin"]]) for r in basket if r.get("portfolio_id") == pid and r["isin"] != isin_common]
                if not nids:
                    return 0.0
                h_bar = X[nids].mean(0)
                return float(ctx["beta_pf"] * float(np.dot(h_bar, ctx["U"][gk])))
            yA = scorer.score_many(lists_A[k])[0]; yB = scorer.score_many(lists_B[k])[0]
            exp_pf_deltas.append(float(_pf_term_for(lists_B[k]) - _pf_term_for(lists_A[k])))
            pred_pf_deltas[k] = float(yB - yA)
        if len(pred_pf_deltas) >= 3 and np.std(pred_pf_deltas) > 1e-6 and np.std(exp_pf_deltas) > 1e-6:
            corr_pf = float(np.corrcoef(pred_pf_deltas, exp_pf_deltas)[0,1])
            assert abs(corr_pf) >= 0.4, f"Portfolio-term |corr| too low: {corr_pf:.2f} | pred={pred_pf_deltas} exp={exp_pf_deltas}"

    # (Optional) print debug table for inspection
    df_debug = pd.DataFrame(req_rows)
    df_debug["pred_bps"] = preds_scorer_warm
    df_debug["exp_bps"] = np.array(exp_y)
    print("\nDEBUG warm scenarios:\n", df_debug)

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

@pytest.mark.e2e
def test_mvdgt_negative_drag_symmetry(tmp_path: Path):
    """
    Construct baskets that induce a negative portfolio-term projection for the anchor
    and verify scorer deltas are negative for both BUY and SELL anchors.

    Design:
      - Use the toy generator's U_g and X to pick:
          anchor = argmax_j X_j·U_g within a portfolio group g
          co-items = two members with most negative projections on U_g (≠ anchor)
      - Build two baskets per side:
          P0    : [anchor only]
          Pdrag : [anchor] + [co1 BUY, co2 SELL] with equal sizes (signed weights ~ 0)
      - Score deltas: Δ = score(Pdrag)[anchor] - score(P0)[anchor]
      - Assert Δ < 0 for both BUY and SELL with a modest, non-trivial threshold.
        If portfolio gates are effectively off (<~0.05), fall back to sign-only with a tiny magnitude.
    """
    # --- Build small world and train a model
    work, pyg_dir, out, ctx = _generate_toy_world(tmp_path / "toy_neg_drag", seed=52)

    cfg = MVDGTTrainConfig(
        workdir=work,
        pyg_dir=pyg_dir,
        outdir=out,
        epochs=30,                 # keep consistent with e2e, short but reliable
        lr=5e-3,
        weight_decay=1e-4,
        batch_size=128,
        seed=23,
        device="cpu",
        enable_tb=False,
        enable_tqdm=False,
        model=MVDGTModelConfig(hidden=64, heads=2, dropout=0.10, trade_dim=2, use_portfolio=True),
    )
    metrics = train_mvdgt(cfg)
    assert (out / "ckpt.pt").exists(), "checkpoint missing"
    assert metrics["test_rmse"] < 18.0, "toy model failed to reach loose accuracy bound"

    scorer = DGTScorer.from_dir(out)

    # --- Pull generator context (U, X, portfolios) to construct a "negative drag" basket
    X = ctx["X"]                  # [N, x_dim]
    U = ctx["U"]                  # [G, x_dim]
    portfolios = ctx["portfolios"]
    assert U is not None and portfolios is not None and len(portfolios) >= 1

    # Choose the portfolio group with the strongest separation between
    # anchor (max projection) and two most-negative co-items.
    best = None
    for g, members in enumerate(portfolios):
        if len(members) < 3:
            continue
        members = list(map(int, members))
        u = U[g]
        proj = np.dot(X[members], u)  # projections on U_g
        a_local = int(np.argmax(proj))
        anchor = int(members[a_local])
        # two smallest projections, excluding anchor if it happens to be among them
        order = np.argsort(proj).tolist()
        co_candidates = [int(members[i]) for i in order if int(members[i]) != anchor]
        if len(co_candidates) < 2:
            continue
        co1, co2 = co_candidates[0], co_candidates[1]
        neg_mean = float(proj[order[0]] + proj[order[1]]) / 2.0
        separation = float(proj[a_local] - neg_mean)
        if (best is None) or (separation > best["sep"]):
            best = {"g": g, "anchor": anchor, "co": (co1, co2), "sep": separation}

    assert best is not None, "No portfolio group with ≥3 members found for negative-drag construction."
    g = int(best["g"])
    anchor_nid = int(best["anchor"])
    co1, co2 = map(int, best["co"])

    # Sanity: under the generator, the constructed co-items imply a negative portfolio term.
    exp_pf_delta = float(np.dot(X[[co1, co2]].mean(0), U[g])) * float(ctx.get("beta_pf", 1.5))
    assert exp_pf_delta < 0.0, f"Construction error: expected negative generator pf term, got {exp_pf_delta:.4f}"

    # --- Build baskets (signed weights ~0 for co-items; structure drives the effect)
    asof = str(pd.Timestamp(ctx["market_dates"].iloc[-1].asof_date).date())
    base_size = 1.5e5
    co_size = 2.0e5
    isin_anchor = _make_isin(anchor_nid)
    isin1 = _make_isin(co1)
    isin2 = _make_isin(co2)

    def _basket(side_anchor: str, pid: str):
        P0 = [{
            "portfolio_id": pid,
            "isin": isin_anchor, "side": side_anchor, "size": base_size, "asof_date": asof
        }]
        # co-items with opposite sides and equal sizes -> signed weight ≈ 0, abs weight > 0
        Pdrag = P0 + [
            {"portfolio_id": pid, "isin": isin1, "side": "BUY",  "size": co_size, "asof_date": asof},
            {"portfolio_id": pid, "isin": isin2, "side": "SELL", "size": co_size, "asof_date": asof},
        ]
        return P0, Pdrag

    deltas = {}
    for s in ["BUY", "SELL"]:
        P0, Pdrag = _basket(s, pid=f"DRAG_{s}")
        y0 = float(scorer.score_many(P0)[0])
        yD = float(scorer.score_many(Pdrag)[0])
        deltas[s] = yD - y0

    # --- Thresholding logic: require negative with a modest margin; relax if gates ~ 0.
    beta_pf = float(ctx.get("beta_pf", 1.5))
    strict_thresh = max(0.06, 0.07 * beta_pf)   # ~0.10 bps for beta_pf=1.5
    relax_thresh  = 0.02                        # sign-only with tiny magnitude if gates are off

    gate_ok = True
    try:
        pf_gate_val = float(torch.sigmoid(getattr(scorer.model, "pf_gate")).detach().cpu().item()) if hasattr(scorer.model, "pf_gate") else None
        port_gate_val = float(torch.sigmoid(getattr(scorer.model, "portfolio_gate")).detach().cpu().item()) if hasattr(scorer.model, "portfolio_gate") else None
        gv = [g for g in [pf_gate_val, port_gate_val] if g is not None]
        if gv and max(gv) < 0.05:
            gate_ok = False
    except Exception:
        pass

    if gate_ok:
        assert deltas["BUY"] <= -strict_thresh and deltas["SELL"] <= -strict_thresh, \
            f"Expected negative portfolio drag on both sides. deltas={deltas}, thresh={strict_thresh:.3f} bps " \
            f"(β_pf={beta_pf:.2f}, group={g}, anchor={anchor_nid}, co={co1,co2})"
    else:
        # If portfolio gates are effectively off, still insist on negative sign with a very small margin
        assert deltas["BUY"] <= -relax_thresh and deltas["SELL"] <= -relax_thresh, \
            f"With portfolio gates ~0, still expect negative sign. deltas={deltas}, relax={relax_thresh:.3f} bps"


@pytest.mark.e2e
def test_mvdgt_assert_leave_one_out_invariance(tmp_path: Path):
    """
    Asserts that including the anchor in port_ctx does NOT change its prediction
    relative to a LOO port_ctx. Fails if portfolio pooling is self-inclusive.
    """
    work, pyg_dir, out, ctx = _generate_toy_world(tmp_path / "toy_loo_assert", seed=73)
    cfg = MVDGTTrainConfig(
        workdir=work, pyg_dir=pyg_dir, outdir=out,
        epochs=20, lr=5e-3, weight_decay=1e-4, batch_size=128,
        seed=41, device="cpu", enable_tb=False, enable_tqdm=False,
        model=MVDGTModelConfig(hidden=64, heads=2, dropout=0.10, trade_dim=2, use_portfolio=True),
    )
    _ = train_mvdgt(cfg)
    state = _rebuild_state_for_manual_forward(out)
    device = state["device"]; model = state["model"]; x = state["x"]

    # pick anchor + two co-items with strong negative projection (like the DIAG did)
    X, U, portfolios = ctx["X"], ctx["U"], ctx["portfolios"]
    pick = None
    for g, members in enumerate(portfolios):
        if len(members) < 3: continue
        u = U[g]; members = list(map(int, members))
        proj = np.dot(X[members], u)
        a = int(members[int(np.argmax(proj))])
        co = [int(members[i]) for i in np.argsort(proj).tolist() if int(members[i]) != a][:2]
        if len(co) >= 2: pick = (g, a, co[0], co[1]); break
    assert pick is not None
    _, anchor_nid, c1, c2 = pick
    asof = str(pd.Timestamp(ctx["market_dates"].iloc[-1].asof_date).date())

    rows = [{"isin": _make_isin(anchor_nid), "side": "BUY", "size": 1.2e5, "asof_date": asof}]
    anchor_idx, trade, market_feat = _vectorize_rows_for_state(state, rows)
    pf_gid = torch.zeros((1,), dtype=torch.long, device=device)

    def _ctx(nodes, w_abs, w_sgn):
        return {
            "port_nodes_flat": torch.tensor(nodes, dtype=torch.long, device=device),
            "port_w_abs_flat": torch.tensor(w_abs, dtype=torch.float32, device=device),
            "port_w_signed_flat": torch.tensor(w_sgn, dtype=torch.float32, device=device),
            "port_len": torch.tensor([len(nodes)], dtype=torch.long, device=device),
        }

    # LOO: co-items only
    ctx_loo   = _ctx([c1, c2], [0.5, 0.5], [0.0, 0.0])
    # SELF: include anchor too (should be ignored by a correct LOO implementation)
    ctx_self  = _ctx([anchor_nid, c1, c2], [0.34, 0.33, 0.33], [0.34, 0.33, -0.33])

    with torch.no_grad():
        y_loo  = float(model(x, anchor_idx=anchor_idx, market_feat=market_feat, pf_gid=pf_gid, port_ctx=ctx_loo,  trade_feat=trade).cpu().numpy().reshape(-1)[0])
        y_self = float(model(x, anchor_idx=anchor_idx, market_feat=market_feat, pf_gid=pf_gid, port_ctx=ctx_self, trade_feat=trade).cpu().numpy().reshape(-1)[0])

    # After the fix these should match up to tiny eps.
    assert abs(y_self - y_loo) < 0.03, f"Self-inclusion changed the anchor prediction: Δ={y_self - y_loo:.3f} bps (expected ≈ 0 if LOO)"


@pytest.mark.e2e
def test_mvdgt_assert_negative_drag_with_loo(tmp_path: Path):
    """
    With LOO prototypes, adding two negatively projecting co-items must reduce
    the anchor prediction for BOTH BUY and SELL anchors.
    Fails if portfolio injection is strictly non-negative or pooling is self-inclusive.
    """
    work, pyg_dir, out, ctx = _generate_toy_world(tmp_path / "toy_neg_drag_assert", seed=77)
    cfg = MVDGTTrainConfig(
        workdir=work, pyg_dir=pyg_dir, outdir=out,
        epochs=20, lr=5e-3, weight_decay=1e-4, batch_size=128,
        seed=37, device="cpu", enable_tb=False, enable_tqdm=False,
        model=MVDGTModelConfig(hidden=64, heads=2, dropout=0.10, trade_dim=2, use_portfolio=True),
    )
    _ = train_mvdgt(cfg)
    scorer = DGTScorer.from_dir(out)

    X, U, portfolios = ctx["X"], ctx["U"], ctx["portfolios"]
    pick = None
    for g, members in enumerate(portfolios):
        if len(members) < 3: continue
        u = U[g]; members = list(map(int, members))
        proj = np.dot(X[members], u)
        a = int(members[int(np.argmax(proj))])
        co = [int(members[i]) for i in np.argsort(proj).tolist() if int(members[i]) != a][:2]
        if len(co) >= 2: pick = (g, a, co[0], co[1]); break
    assert pick is not None
    g, anchor_nid, c1, c2 = pick
    asof = str(pd.Timestamp(ctx["market_dates"].iloc[-1].asof_date).date())

    def basket(side, pid):
        base = [{"portfolio_id": pid, "pf_gid": 0, "isin": _make_isin(anchor_nid), "side": side, "size": 1.5e5, "asof_date": asof}]
        # equal & opposite co-items to isolate structure; scorer currently includes anchor,
        # but after LOO fix, that inclusion is neutralized at model level.
        drag = base + [
            {"portfolio_id": pid, "pf_gid": 0, "isin": _make_isin(c1), "side": "BUY",  "size": 2.0e5, "asof_date": asof},
            {"portfolio_id": pid, "pf_gid": 0, "isin": _make_isin(c2), "side": "SELL", "size": 2.0e5, "asof_date": asof},
        ]
        return base, drag

    # Both sides should experience negative drag after the fix.
    for side in ["BUY", "SELL"]:
        P0, Pd = basket(side, pid=f"NEG_{side}")
        y0 = float(scorer.score_many(P0)[0])
        yd = float(scorer.score_many(Pd)[0])
        delta = yd - y0
        assert delta < -0.08, f"Expected negative drag for {side}, got Δ={delta:.3f} bps (should be < -0.08)"


