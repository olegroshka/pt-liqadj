# ptliq/paper/experiment.py
from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from ptliq.service.scoring import DGTScorer
from ptliq.model.mv_dgt import MultiViewDGT
from ptliq.training.mvdgt_loop import MVDGTModelConfig
from ptliq.utils.logging_utils import get_logger


# -----------------------------
# Small utilities
# -----------------------------
def _run(cmd: List[str], cwd: Optional[Path] = None) -> None:
    """Run a command with live output and fail fast."""
    print(">>", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _to_date(x: Any):
    if x is None:
        return None
    try:
        return pd.to_datetime(x, errors="coerce").normalize()
    except Exception:
        return None

def _side_sign(x: Any) -> float:
    if x is None:
        return 0.0
    s = str(x).strip().upper()
    if s in {"B","BUY","CBUY","TRUE","1"}:
        return 1.0
    if s in {"S","SELL","CSELL","FALSE","0","-1"}:
        return -1.0
    return 0.0

def _log_size(x: Any) -> float:
    try:
        v = float(x); return float(math.log1p(abs(v))) if v == v else 0.0
    except Exception:
        return 0.0


# -----------------------------
# Config for the orchestration
# -----------------------------
@dataclass
class MakeDataConfig:
    """
    All paths are folders. We only *call* existing CLIs; no duplication of logic.
    """
    root: Path                           # base working directory (e.g., Path("paper_runs/exp001"))
    seed: int = 42                       # forwarded to simulate/train when supported
    n_nodes: Optional[int] = None        # forwarded if your simulate supports it
    n_days: Optional[int] = None         # forwarded if your simulate supports it
    out_model_dir: Optional[Path] = None # defaults to root / "models" / "dgt"
    overwrite: bool = True               # delete old subfolders if present

    # Extra knobs forwarded to CLIs (keep it open-ended)
    simulate_args: Tuple[str, ...] = ()
    featurize_graph_args: Tuple[str, ...] = ()
    featurize_pyg_args: Tuple[str, ...] = ()
    dgt_build_args: Tuple[str, ...] = ()
    dgt_train_args: Tuple[str, ...] = ()


# -----------------------------
# Orchestration: data -> build -> train
# -----------------------------
def make_data(cfg: MakeDataConfig) -> Dict[str, str]:
    """
    Pipeline using your existing CLIs:

      ptliq-simulate
      ptliq-featurize graph ...
      ptliq-featurize pyg ...
      ptliq-dgt-build ...
      ptliq-dgt-train ...

    Returns a dict of resolved paths (for recording into paper_meta.json).
    """
    root = Path(cfg.root).resolve()
    if cfg.overwrite and root.exists():
        # don't delete the whole root; only our standard subfolders
        for sub in ["data/raw/sim", "data/graph", "data/pyg", "data/mvdgt", "models"]:
            p = root / sub
            if p.exists():
                print(f"[clean] rm -rf {p}")
                import shutil; shutil.rmtree(p, ignore_errors=True)

    raw_dir = _ensure_dir(root / "data/raw/sim")
    graph_dir = root / "data/graph"
    pyg_dir = root / "data/pyg"
    work_dir = root / "data/mvdgt/exp001"
    model_dir = cfg.out_model_dir or (root / "models/dgt")
    _ensure_dir(graph_dir); _ensure_dir(pyg_dir); _ensure_dir(work_dir); _ensure_dir(model_dir)

    # 1) simulate (your CLI)
    sim_cmd = ["ptliq-simulate", "--outdir", str(raw_dir), "--seed", str(cfg.seed)]
    if cfg.n_nodes is not None:
        sim_cmd += ["--n-nodes", str(cfg.n_nodes)]
    if cfg.n_days is not None:
        sim_cmd += ["--n-days", str(cfg.n_days)]
    sim_cmd += list(cfg.simulate_args)
    _run(sim_cmd)

    # Expect simulate to write bonds + trades parquet
    bonds = raw_dir / "bonds.parquet"
    trades = raw_dir / "trades.parquet"
    assert bonds.exists() and trades.exists(), "simulate should produce bonds.parquet and trades.parquet"

    # 2) featurize -> graph
    _run([
        "ptliq-featurize", "graph",
        "--bonds", str(bonds),
        "--trades", str(trades),
        "--outdir", str(graph_dir),
        *cfg.featurize_graph_args
    ])

    # 3) featurize -> pyg
    _run([
        "ptliq-featurize", "pyg",
        "--graph-dir", str(graph_dir),
        "--outdir", str(pyg_dir),
        *cfg.featurize_pyg_args
    ])

    # 4) dgt build (makes mvdgt_meta.json + samples etc under work_dir)
    _run([
        "ptliq-dgt-build",
        "--trades-path", str(trades),
        "--graph-dir", str(graph_dir),
        "--pyg-dir", str(pyg_dir),
        "--outdir", str(work_dir),
        *cfg.dgt_build_args
    ])

    # 5) dgt train (saves ckpt + configs under model_dir)
    _run([
        "ptliq-dgt-train",
        "--workdir", str(work_dir),
        "--pyg-dir", str(pyg_dir),
        "--epochs", "30",
        "--lr", "5e-3",
        "--batch-size", "512",
        "--seed", str(cfg.seed),
        "--device", "auto",
        "--outdir", str(model_dir),
        *cfg.dgt_train_args
    ])

    # Record meta to tie artifacts together
    paper_meta = {
        "root": str(root),
        "raw_dir": str(raw_dir),
        "graph_dir": str(graph_dir),
        "pyg_dir": str(pyg_dir),
        "work_dir": str(work_dir),
        "model_dir": str(model_dir),
    }
    (root / "paper_meta.json").write_text(json.dumps(paper_meta, indent=2))
    return paper_meta


# -----------------------------
# Scenario scoring (A/B/C/D, warm/cold, parity, drift, ablation)
# -----------------------------
def _load_run_artifacts(model_dir: Path):
    model_dir = Path(model_dir)
    meta = json.loads((model_dir / "mvdgt_meta.json").read_text())
    files = meta.get("files", {})
    # graph + contexts
    pyg_graph = Path(files["pyg_graph"])
    data = torch.load(pyg_graph, map_location="cpu", weights_only=False)
    x = data.x.float()
    edge_index = data.edge_index
    edge_weight = data.edge_weight if hasattr(data, "edge_weight") else None
    # samples (training copy should be in outdir)
    # prefer meta pointer, else sibling file
    samples_path = files.get("samples") or str((model_dir / "samples.parquet"))
    if not Path(samples_path).exists():
        raise FileNotFoundError("samples.parquet not found; ensure training persisted a copy and meta['files']['samples'] points to it.")
    samples = pd.read_parquet(samples_path)

    # nodes map
    nodes_path = files.get("graph_nodes")
    if not nodes_path:
        # Fallbacks around the pyg graph location
        candidates = []
        if pyg_graph:
            # 1) Same folder as pyg_graph
            candidates.append(Path(pyg_graph).parent / "graph_nodes.parquet")
            # 2) Sibling graph folder next to pyg folder (…/data/graph/graph_nodes.parquet)
            candidates.append(Path(pyg_graph).parent.parent / "graph" / "graph_nodes.parquet")
        # Pick the first existing candidate
        for c in candidates:
            if c.exists():
                nodes_path = str(c)
                break
        if not nodes_path:
            raise FileNotFoundError(
                "graph_nodes.parquet not found; looked next to pyg graph and in sibling 'graph' folder."
            )
    nodes = pd.read_parquet(nodes_path)
    nid2isin = {int(r.node_id): str(r.isin) for r in nodes.itertuples(index=False)}
    isin2nid = {v: k for k, v in nid2isin.items()}

    # masks path (for ablation convenience)
    masks_path = files.get("view_masks") or str((model_dir / "view_masks.pt"))

    return {
        "meta": meta,
        "samples": samples,
        "x": x,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "nid2isin": nid2isin,
        "isin2nid": isin2nid,
        "masks_path": masks_path,
    }


def _direct_forward(
    model_dir: Path,
    req_rows: List[Dict[str, Any]],
    view_masks_path: Optional[Path] = None,
    device: str = "auto",
) -> np.ndarray:
    """
    Direct forward using MultiViewDGT, mirroring DGTScorer’s preprocessing.
    Kept nearly identical to your e2e test _direct_predict() for parity.
    """
    # device auto-selection
    if device == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(device)
    # unify naming for subsequent code
    device = dev

    ckpt = torch.load(Path(model_dir) / "ckpt.pt", map_location="cpu")
    # Prefer persisted model_config.json when present; fallback to checkpoint's embedded config
    cfg_dict = None
    json_cfg_path = Path(model_dir) / "model_config.json"
    if json_cfg_path.exists():
        cfg_dict = json.loads(json_cfg_path.read_text())
    else:
        cfg_dict = ckpt.get("model_config", {})
    model_cfg = MVDGTModelConfig(**cfg_dict)
    meta = json.loads((Path(model_dir) / "mvdgt_meta.json").read_text())
    vm_path = Path(model_dir) / "view_masks.pt" if (view_masks_path is None) else Path(view_masks_path)
    view_masks = torch.load(vm_path, map_location="cpu")

    # graph + contexts
    data = torch.load(Path(meta["files"]["pyg_graph"]), map_location="cpu", weights_only=False)
    x = data.x.float().to(device)
    edge_index = data.edge_index.to(device)
    edge_weight = data.edge_weight.to(device) if hasattr(data, "edge_weight") else None
    vm = {k: v.to(device) for k, v in view_masks.items()}

    # market (z-score + sign if available)
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
    # market preproc (saved by training)
    mkt_mean = None; mkt_std = None; mkt_sign = 1.0
    preproc_path = Path(model_dir) / "market_preproc.json"
    if preproc_path.exists():
        pre = json.loads(preproc_path.read_text())
        mkt_mean = torch.tensor(pre.get("mean", []), dtype=torch.float32, device=device)
        std_vals = [float(v) if (v is not None and float(v) > 0.0) else 1.0 for v in pre.get("std", [])]
        mkt_std = torch.tensor(std_vals, dtype=torch.float32, device=device)
        mkt_sign = float(pre.get("sign", 1.0))

    # nodes
    files = meta.get("files", {})
    nodes_path = files.get("graph_nodes")
    if not nodes_path:
        pyg_graph = Path(files.get("pyg_graph", ""))
        candidates = []
        if pyg_graph:
            candidates.append(Path(pyg_graph).parent / "graph_nodes.parquet")
            candidates.append(Path(pyg_graph).parent.parent / "graph" / "graph_nodes.parquet")
        for c in candidates:
            if c.exists():
                nodes_path = str(c)
                break
        if not nodes_path:
            raise FileNotFoundError(
                "graph_nodes.parquet not found; looked next to pyg graph and in sibling 'graph' folder."
            )
    nodes = pd.read_parquet(nodes_path)
    isin_to_node = {str(r.isin): int(r.node_id) for r in nodes.itertuples(index=False)}

    # scaler + feature order
    fnames = json.loads((Path(model_dir) / "feature_names.json").read_text())
    sc = json.loads((Path(model_dir) / "scaler.json").read_text())
    mean = torch.tensor([float(sc["mean"][i]) for i in range(len(fnames))], dtype=torch.float32, device=device)
    std  = torch.tensor([float(sc["std"][i]) if (float(sc["std"][i]) > 0.0) else 1.0
                         for i in range(len(fnames))], dtype=torch.float32, device=device)

    # model (mirror DGTScorer wiring; honor saved model_config.json)
    # Ensure market dims/flags if missing in saved config
    if getattr(model_cfg, "mkt_dim", None) is None:
        try:
            model_cfg.mkt_dim = int(mkt_ctx["mkt_feat"].size(1)) if (mkt_ctx is not None) else 0
        except Exception:
            model_cfg.mkt_dim = 0
    if getattr(model_cfg, "use_market", None) is None:
        try:
            model_cfg.use_market = bool(int(model_cfg.mkt_dim) > 0)
        except Exception:
            model_cfg.use_market = False

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
        view_names=list(getattr(model_cfg, "views", ["struct", "port", "corr_global", "corr_local"])),
        use_pf_head=bool(getattr(model_cfg, "use_pf_head", False)),
        pf_head_hidden=getattr(model_cfg, "pf_head_hidden", None),
        # portfolio attention wiring
        use_portfolio_attn=bool(getattr(model_cfg, "use_portfolio_attn", False)),
        portfolio_attn_layers=int(getattr(model_cfg, "portfolio_attn_layers", 1) or 1),
        portfolio_attn_heads=int(getattr(model_cfg, "portfolio_attn_heads", 4) or 4),
        portfolio_attn_dropout=float(getattr(model_cfg, "portfolio_attn_dropout", model_cfg.dropout if getattr(model_cfg, "dropout", None) is not None else 0.1) or 0.1),
        portfolio_attn_hidden=(int(getattr(model_cfg, "portfolio_attn_hidden")) if (getattr(model_cfg, "portfolio_attn_hidden", None) is not None) else None),
        portfolio_attn_concat_trade=bool(getattr(model_cfg, "portfolio_attn_concat_trade", True)),
        portfolio_attn_concat_market=bool(getattr(model_cfg, "portfolio_attn_concat_market", False)),
        portfolio_attn_mode=str(getattr(model_cfg, "portfolio_attn_mode", "residual")),
        portfolio_attn_gate_init=float(getattr(model_cfg, "portfolio_attn_gate_init", 0.0) or 0.0),
        max_portfolio_len=(int(getattr(model_cfg, "max_portfolio_len")) if (getattr(model_cfg, "max_portfolio_len", None) is not None) else None),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    # vectorize rows
    def _vec(rows: List[Dict[str, Any]]):
        node_ids = []
        for r in rows:
            isin = str(r.get("isin","")).strip()
            if isin not in isin_to_node:
                raise ValueError(f"Unknown ISIN: {isin}")
            node_ids.append(isin_to_node[isin])
        anchor_idx = torch.as_tensor(node_ids, dtype=torch.long, device=device)

        # trade
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

        # market
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
                market_feat = torch.nan_to_num(market_feat, nan=0.0, posinf=0.0, neginf=0.0) * float(mkt_sign)

        return anchor_idx, trade, market_feat

    anchor_idx, trade, market_feat = _vec(req_rows)
    with torch.no_grad():
        yhat = model(x, anchor_idx=anchor_idx, market_feat=market_feat,
                     pf_gid=None, port_ctx=None, trade_feat=trade)
        y = yhat.detach().cpu().numpy().astype(np.float32).reshape(-1)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y


def score_scenarios(
    model_dir: Path,
    out_dir: Path,
    n_warm_groups: int = 3,
    reps_per_group: int = 2,
    seed: int = 100,
) -> Dict[str, str]:
    """
    Replays the e2e scenarios and writes multiple CSVs:
      - warm_scenarios.csv (A/B/C/D)
      - cold_scenarios.csv (A/B/C)
      - portfolio_drift.csv (A vs B baskets across anchors)
      - ablation.csv (mask port view & no portfolio context)
      - parity.csv (scorer vs direct)
      - negative_drag.csv (BUY/SELL deltas constructed to be negative under LOO)
    Returns paths for convenience.
    """
    rng = np.random.default_rng(seed)
    out_dir = _ensure_dir(Path(out_dir))

    # logger and device selection
    logger = get_logger("ptliq.paper.score_scenarios", out_dir, filename="score_scenarios.log")
    if torch.cuda.is_available():
        device_str = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        device_str = "mps"
    else:
        device_str = "cpu"
    logger.info(f"score_scenarios: using device={device_str}")

    # Load artifacts
    logger.info("Loading run artifacts…")
    art = _load_run_artifacts(Path(model_dir))
    logger.info("Loaded artifacts: samples=%d, masks_path=%s", len(art["samples"]), art["masks_path"]) 
    samples: pd.DataFrame = art["samples"]
    nid2isin = art["nid2isin"]
    isin2nid = art["isin2nid"]
    masks_path = Path(art["masks_path"])

    # Scorer
    logger.info("Initializing DGTScorer…")
    scorer = DGTScorer.from_dir(str(model_dir), device=device_str)
    logger.info("DGTScorer ready on device=%s", device_str)

    # Helper: pick warm reps (groups present in train)
    tr_pf = samples[(samples["split"] == "train") & (samples["pf_gid"] >= 0)]
    warm_groups = sorted(tr_pf["pf_gid"].unique().tolist())[:max(1, n_warm_groups)]
    reps = []
    for g in warm_groups:
        grp = tr_pf[tr_pf["pf_gid"] == g]
        if len(grp) == 0:
            continue
        idx = rng.choice(grp.index.to_numpy(), size=min(reps_per_group, len(grp)), replace=False)
        reps.extend([grp.loc[i] for i in idx])

    # Helper: pick cold reps (groups not in train)
    all_groups = sorted(samples["pf_gid"].dropna().astype(int).unique().tolist())
    cold_groups = [g for g in all_groups if g not in warm_groups]
    cold_reps = []
    if len(cold_groups) > 0:
        cg = cold_groups[-1]
        src = samples[(samples["pf_gid"] == cg) & (samples["split"] != "train")]
        if len(src) > 0:
            idx = rng.choice(src.index.to_numpy(), size=min(2, len(src)), replace=False)
            cold_reps = [src.loc[i] for i in idx]

    # Shortcuts for reading market index if present
    meta = art["meta"]
    mi_path = meta.get("files", {}).get("market_index")
    mi = None
    if mi_path and Path(mi_path).exists():
        mi = pd.read_parquet(mi_path)
        mi["asof_date"] = pd.to_datetime(mi["asof_date"]).dt.normalize()

    def _asof_for_idx(di: int) -> str:
        if mi is None:
            return ""
        row = mi[mi["row_idx"] == int(di)]
        if row.empty:
            return ""
        return str(pd.Timestamp(row.iloc[0].asof_date).date())

    # ---------- Warm: A/B/C/D scenarios ----------
    warm_rows: List[Dict[str, Any]] = []
    for r in reps:
        nid = int(r.node_id)
        asin = nid2isin[nid]
        di = int(r.date_idx)
        asof = _asof_for_idx(di)
        base_size = float(np.expm1(float(r.log_size)))
        base_side = "BUY" if float(r.side_sign) > 0 else "SELL"

        # A: identical
        warm_rows.append({"portfolio_id": f"PF-{int(r.pf_gid)}", "pf_gid": int(r.pf_gid),
                          "isin": asin, "side": base_side, "size": base_size, "asof_date": asof})
        # B: +20% size
        warm_rows.append({"portfolio_id": f"PF-{int(r.pf_gid)}", "pf_gid": int(r.pf_gid),
                          "isin": asin, "side": base_side, "size": base_size * 1.2, "asof_date": asof})
        # C: flip side
        warm_rows.append({"portfolio_id": f"PF-{int(r.pf_gid)}", "pf_gid": int(r.pf_gid),
                          "isin": asin, "side": ("SELL" if base_side == "BUY" else "BUY"), "size": base_size, "asof_date": asof})
        # D: next day (if market index)
        if mi is not None:
            di2 = min(di + 1, int(mi["row_idx"].max()))
            warm_rows.append({"portfolio_id": f"PF-{int(r.pf_gid)}", "pf_gid": int(r.pf_gid),
                              "isin": asin, "side": base_side, "size": base_size, "asof_date": _asof_for_idx(di2)})
        else:
            warm_rows.append({"portfolio_id": f"PF-{int(r.pf_gid)}", "pf_gid": int(r.pf_gid),
                              "isin": asin, "side": base_side, "size": base_size})

    logger.info("Scoring warm scenarios (%d rows)…", len(warm_rows))
    y_scorer_warm = scorer.score_many(warm_rows)
    y_direct_warm = _direct_forward(Path(model_dir), warm_rows, device=device_str)
    scen_labels = ["A","B","C","D"] * (len(warm_rows)//4) + (["A"] * (len(warm_rows) % 4))
    df_warm = pd.DataFrame(warm_rows)
    df_warm["scenario"] = scen_labels[:len(df_warm)]
    df_warm["pred_scorer_bps"] = y_scorer_warm
    df_warm["pred_direct_bps"] = y_direct_warm
    warm_path = out_dir / "warm_scenarios.csv"
    df_warm.to_csv(warm_path, index=False)
    logger.info("Wrote %s", warm_path)

    # ---------- Cold: A/B/C ----------
    cold_rows: List[Dict[str, Any]] = []
    for r in cold_reps:
        nid = int(r.node_id)
        asin = nid2isin[nid]
        di = int(r.date_idx)
        asof = _asof_for_idx(di)
        base_size = float(np.expm1(float(r.log_size)))
        base_side = "BUY" if float(r.side_sign) > 0 else "SELL"
        # A:
        cold_rows.append({"portfolio_id": f"PF-{int(r.pf_gid)}", "pf_gid": int(r.pf_gid),
                          "isin": asin, "side": base_side, "size": base_size, "asof_date": asof})
        # B:
        cold_rows.append({"portfolio_id": f"PF-{int(r.pf_gid)}", "pf_gid": int(r.pf_gid),
                          "isin": asin, "side": base_side, "size": base_size * 1.5, "asof_date": asof})
        # C:
        cold_rows.append({"portfolio_id": f"PF-{int(r.pf_gid)}", "pf_gid": int(r.pf_gid),
                          "isin": asin, "side": ("SELL" if base_side == "BUY" else "BUY"),
                          "size": base_size, "asof_date": asof})
    if cold_rows:
        logger.info("Scoring cold scenarios (%d rows)…", len(cold_rows))
        y_scorer_cold = scorer.score_many(cold_rows)
        y_direct_cold = _direct_forward(Path(model_dir), cold_rows, device=device_str)
        df_cold = pd.DataFrame(cold_rows)
        df_cold["scenario"] = ["A","B","C"] * (len(cold_rows)//3) + (["A"] * (len(cold_rows) % 3))
        df_cold["pred_scorer_bps"] = y_scorer_cold
        df_cold["pred_direct_bps"] = y_direct_cold
        cold_path = out_dir / "cold_scenarios.csv"
        df_cold.to_csv(cold_path, index=False)
        logger.info("Wrote %s", cold_path)

    # ---------- Portfolio drift: A vs B baskets per anchor (warm reps) ----------
    # Build two baskets per anchor: P_A = anchor + 2 same‑group BUY co‑items
    # P_B = anchor + 2 different‑group SELL co‑items at larger sizes
    abs_deltas = []
    rows_A = []
    rows_B = []

    portfolios_by_gid: Dict[int, List[int]] = {}
    for gid in samples["pf_gid"].dropna().astype(int).unique().tolist():
        members = samples[samples["pf_gid"] == gid]["node_id"].astype(int).unique().tolist()
        portfolios_by_gid[int(gid)] = members

    K = min(6, len(reps))
    for k in range(K):
        r = reps[k]
        nid_k = int(r.node_id)
        asin_k = art["nid2isin"][nid_k]
        asof_k = _asof_for_idx(int(r.date_idx))
        base_size_k = float(np.expm1(float(r.log_size)))
        side_lbl = "BUY" if float(r.side_sign) > 0 else "SELL"
        gk = int(r.pf_gid)

        group_members = [m for m in portfolios_by_gid.get(gk, []) if m != nid_k]
        others1 = group_members[:2] if len(group_members) >= 2 else [j for j in portfolios_by_gid.get(gk, []) if j != nid_k][:2]
        # choose a different group deterministically
        other_groups = [g for g in portfolios_by_gid.keys() if g != gk]
        others2 = (portfolios_by_gid[other_groups[0]][:2] if other_groups else others1)

        P_A = [{"portfolio_id": f"A_{k}", "pf_gid": 0, "isin": asin_k, "side": side_lbl,
                "size": base_size_k, "asof_date": asof_k}] + \
              [{"portfolio_id": f"A_{k}", "pf_gid": 0, "isin": art["nid2isin"][j], "side": "BUY",
                "size": 1.0e5, "asof_date": asof_k} for j in others1]

        P_B = [{"portfolio_id": f"B_{k}", "pf_gid": 1, "isin": asin_k, "side": side_lbl,
                "size": base_size_k, "asof_date": asof_k}] + \
              [{"portfolio_id": f"B_{k}", "pf_gid": 1, "isin": art["nid2isin"][j], "side": "SELL",
                "size": 2.0e5, "asof_date": asof_k} for j in others2]

        rows_A.append(P_A)
        rows_B.append(P_B)

    logger.info("Computing portfolio drift across %d anchors…", len(rows_A))
    deltas = []
    for P_A, P_B in zip(rows_A, rows_B):
        yA = float(scorer.score_many(P_A)[0])
        yB = float(scorer.score_many(P_B)[0])
        deltas.append(yB - yA)
        abs_deltas.append(abs(yB - yA))
    df_drift = pd.DataFrame({"delta_bps": deltas, "abs_delta_bps": abs_deltas})
    drift_path = out_dir / "portfolio_drift.csv"
    df_drift.to_csv(drift_path, index=False)
    try:
        logger.info("Wrote %s (avg |delta|=%.3f bps)", drift_path, float(np.mean(abs_deltas)) if abs_deltas else float('nan'))
    except Exception:
        logger.info("Wrote %s", drift_path)

    # ---------- View ablation + context removal on the strongest drift pair ----------
    if rows_A:
        logger.info("Running ablation: zeroing 'port' view and disabling portfolio context…")
        # Save a masks copy with 'port' view zeroed
        vm = torch.load(masks_path, map_location="cpu")
        vm_ablate = {k: v.clone() for k, v in vm.items()}
        if "port" in vm_ablate:
            vm_ablate["port"] = torch.zeros_like(vm_ablate["port"])
        ablate_path = Path(model_dir) / "view_masks_port_off.pt"
        torch.save(vm_ablate, ablate_path)
        logger.info("Ablation masks saved to %s", ablate_path)

        idx_best = int(np.argmax(np.asarray(abs_deltas)))
        from_rowsA = rows_A[idx_best]
        from_rowsB = rows_B[idx_best]

        # parity with full model (direct path uses masks under model_dir)
        yA_full = float(scorer.score_many(from_rowsA)[0])
        yB_full = float(scorer.score_many(from_rowsB)[0])

        # direct forward with 'port' mask off and *portfolio context disabled*
        def _strip_pf(rows):
            return [{k:v for k,v in r.items() if k not in ("portfolio_id","pf_gid")} for r in rows]
        yA_port0 = float(_direct_forward(Path(model_dir), _strip_pf(from_rowsA), view_masks_path=ablate_path, device=device_str)[0])
        yB_port0 = float(_direct_forward(Path(model_dir), _strip_pf(from_rowsB), view_masks_path=ablate_path, device=device_str)[0])

        df_ablate = pd.DataFrame([{
            "delta_full_bps": yB_full - yA_full,
            "delta_mask0_nopf_bps": yB_port0 - yA_port0,
        }])
        ablate_out = out_dir / "ablation.csv"
        df_ablate.to_csv(ablate_out, index=False)
        logger.info("Wrote %s", ablate_out)

    # ---------- BUY/SELL negative drag (sign‑symmetric) ----------
    # Build baskets: P0 = [anchor only]; Pdrag = add 2 co‑items with equal & opposite sizes → signed weight ~ 0
    # Drag must be negative for BUY and SELL if the portfolio residual is sign‑agnostic LOO.
    neg_rows = []
    sides = ["BUY", "SELL"]
    # choose a consistent anchor from reps if available
    if reps:
        r = reps[0]
        anchor_nid = int(r.node_id)
        asin_anchor = art["nid2isin"][anchor_nid]
        asof = _asof_for_idx(int(r.date_idx))
        # pick two co‑items from the same group (or arbitrary)
        gk = int(r.pf_gid)
        group_members = [m for m in portfolios_by_gid.get(gk, []) if m != anchor_nid]
        if len(group_members) < 2:
            # fallback: any two nodes ≠ anchor
            pool = [nid for nid in art["nid2isin"].keys() if nid != anchor_nid]
            group_members = pool[:2]
        if len(group_members) >= 2:
            c1, c2 = int(group_members[0]), int(group_members[1])
            for side in sides:
                P0 = [{"portfolio_id": f"NEG_{side}", "pf_gid": 0, "isin": asin_anchor,
                       "side": side, "size": 1.5e5, "asof_date": asof}]
                Pdrag = P0 + [
                    {"portfolio_id": f"NEG_{side}", "pf_gid": 0, "isin": art["nid2isin"][c1],
                     "side": "BUY", "size": 2.0e5, "asof_date": asof},
                    {"portfolio_id": f"NEG_{side}", "pf_gid": 0, "isin": art["nid2isin"][c2],
                     "side": "SELL", "size": 2.0e5, "asof_date": asof},
                ]
                y0 = float(scorer.score_many(P0)[0])
                yD = float(scorer.score_many(Pdrag)[0])
                neg_rows.append({"side": side, "delta_bps": yD - y0})
    if neg_rows:
        neg_path = out_dir / "negative_drag.csv"
        pd.DataFrame(neg_rows).to_csv(neg_path, index=False)
        logger.info("Wrote %s", neg_path)

    # ---------- Tiny parity snapshot ----------
    one = [{"isin": list(isin2nid.keys())[0], "side": "BUY", "size": 1e5}]
    p_s = scorer.score_many(one)[0]
    p_d = _direct_forward(Path(model_dir), one, device=device_str)[0]
    parity_path = out_dir / "parity.csv"
    pd.DataFrame([{"pred_scorer_bps": p_s, "pred_direct_bps": p_d, "abs_diff": abs(p_s - p_d)}]) \
      .to_csv(parity_path, index=False)
    logger.info("Wrote %s", parity_path)

    return {
        "warm_scenarios": str(out_dir / "warm_scenarios.csv"),
        "cold_scenarios": str(out_dir / "cold_scenarios.csv"),
        "portfolio_drift": str(out_dir / "portfolio_drift.csv"),
        "ablation": str(out_dir / "ablation.csv"),
        "negative_drag": str(out_dir / "negative_drag.csv"),
        "parity": str(out_dir / "parity.csv"),
    }
