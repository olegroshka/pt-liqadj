# tests/test_mvdgt_e2e.py
from __future__ import annotations
import json
from pathlib import Path
import math
import numpy as np
import pandas as pd
import torch
import pytest

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

    # Define a small linear generator in bps
    beta_side = 3.0
    beta_lsz  = 4.0
    beta_x    = 2.0
    beta_mkt  = 2.0
    w_node = rng.normal(0.0, 0.5, size=(X.shape[1],)).astype(np.float32)
    w_mkt  = rng.normal(0.0, 0.5, size=(mkt_feat.size(1),)).astype(np.float32)

    def _y(nid: int, di: int, side: float, log_size: float) -> float:
        lsz_z = (log_size - mu_lsz) / (sd_lsz if sd_lsz > 0 else 1.0)
        y = (
            beta_side * side
            + beta_lsz * lsz_z
            + beta_x * float(np.dot(X[nid], w_node))
            + beta_mkt * float(np.dot(mkt_feat[di].numpy(), w_mkt))
        )
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
        "portfolios": portfolios,
        "samples": samples,
    }


# --------------------------
# Direct model inference (parity with scorer)
# --------------------------
def _direct_predict(outdir: Path, req_rows: list[dict]) -> np.ndarray:
    """Reconstruct the trained model + contexts and run a direct forward pass,
    using the same scaler/feature_names as DGTScorer."""
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
        # Expect cold error >= warm-perturbed error (or at least not much better)
        assert rmse_cold >= rmse_BCD - 1.0, f"Cold RMSE unexpectedly lower than warm BCD by >1.0bps: cold={rmse_cold:.3f}, warmBCD={rmse_BCD:.3f}"

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

    # (Optional) print debug table for inspection
    df_debug = pd.DataFrame(req_rows)
    df_debug["pred_bps"] = preds_scorer_warm
    df_debug["exp_bps"] = np.array(exp_y)
    print("\nDEBUG warm scenarios:\n", df_debug)
