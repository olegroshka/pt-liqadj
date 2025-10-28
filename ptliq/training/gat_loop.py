# ptliq/training/gat_loop.py
from __future__ import annotations

import json, re
import copy
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import Data
import math
from torch.optim.lr_scheduler import LambdaLR

# Optional progress bar
try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None  # type: ignore

# Optional TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:  # pragma: no cover - tensorboard is optional
    SummaryWriter = None  # type: ignore

from ptliq.model.model import LiquidityModelGAT
from ptliq.model.losses import composite_loss

# ---------------------------
# EMA helper for stability
# ---------------------------
class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.995):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = float(decay)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self.decay
        for p_ema, p in zip(self.ema.parameters(), model.parameters()):
            if p.requires_grad:
                p_ema.data.mul_(d).add_(p.data, alpha=1.0 - d)

    def to(self, device: torch.device) -> "ModelEMA":
        self.ema.to(device)
        return self

# ---------------------------
# Configs
# ---------------------------
@dataclass
class SamplerConfig:
    expo_scale: float = 1e5     # DV01 fraction scale (per-line weight)
    clip_w: float = 5.0         # cap |weight|
    chunk_min: int = 150
    chunk_max: int = 300
    bucket_minutes: int = 15
    seed: int = 17
    coalesce_pf_base: bool = False      # off by default to avoid synthetic baskets
    weight_mode: str = "signed_frac"    # "signed_frac" | "raw_scaled"

@dataclass
class TrainConfig:
    device: str = "auto"
    max_epochs: int = 20
    batch_size: int = 32            # number of baskets per step
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 4
    seed: int = 42
    clip_grad: float = 1.0
    print_every: int = 50
    log_level: str = "INFO"
    enable_tb: bool = True          # enable TensorBoard logging when available
    tb_log_dir: Optional[str] = None  # default to <outdir>/tb if None

@dataclass
class ModelConfig:
    d_model: int = 128
    heads: int = 4
    rel_emb_dim: int = 16
    issuer_emb_dim: int = 16
    dropout: float = 0.10
    rel_init_boost: Dict[int, float] | None = None   # {relation_id: gain}
    encoder_type: str = "gat"  # 'gat' or 'gat_diff'

@dataclass
class LiquidityRunConfig:
    sampler: SamplerConfig
    train: TrainConfig
    model: ModelConfig

# ---------------------------
# Utilities
# ---------------------------
def _device(s: str) -> torch.device:
    if s.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)

def _safe_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

# ---------------------------
# PyG Data loader (robust)
# ---------------------------
def _one_hot(series: pd.Series) -> np.ndarray:
    s = series.fillna("__NA__").astype(str)
    classes = sorted(s.unique().tolist())
    idx = {c: i for i, c in enumerate(classes)}
    X = np.zeros((len(s), len(classes)), dtype=np.float32)
    X[np.arange(len(s)), s.map(idx).values] = 1.0
    return X

def _rebuild_data_from_tabular(nodes_path: Path, edges_path: Path) -> Data:
    # CSV or PARQUET accepted
    if nodes_path.suffix == ".parquet":
        nodes = pd.read_parquet(nodes_path)
    else:
        nodes = pd.read_csv(nodes_path)
    if edges_path.suffix == ".parquet":
        edges = pd.read_parquet(edges_path)
    else:
        edges = pd.read_csv(edges_path)

    N = len(nodes)
    def z(col):
        if col not in nodes: return np.zeros((N, 1), dtype=np.float32)
        x = pd.to_numeric(nodes[col], errors="coerce").astype(float).values
        mu, sd = np.nanmean(x), np.nanstd(x) + 1e-8
        return ((x - mu) / sd).reshape(-1, 1).astype(np.float32)

    zr = z("rating_num")
    zt = z("tenor_years")
    oh_sector = _one_hot(nodes["sector_id"].astype(str) if "sector_id" in nodes else pd.Series(["0"]*N))
    oh_curve  = _one_hot(nodes["curve_bucket"] if "curve_bucket" in nodes else pd.Series(["UNK"]*N))
    oh_ccy    = _one_hot(nodes["currency"] if "currency" in nodes else pd.Series(["USD"]*N))
    x_np = np.hstack([zr, zt, oh_sector, oh_curve, oh_ccy]).astype(np.float32)

    issuer_index = nodes.get("issuer_id", pd.Series([-1]*N)).fillna(-1).astype(int).values

    src = edges["src_id"].astype(int).values
    dst = edges["dst_id"].astype(int).values
    ety = edges["relation_id"].astype(int).values
    ewt = pd.to_numeric(edges["weight"], errors="coerce").fillna(0.0).astype(float).values

    # make directed
    edge_index = np.vstack([np.concatenate([src, dst]), np.concatenate([dst, src])]).astype(np.int64)
    edge_type  = np.concatenate([ety, ety]).astype(np.int64)
    edge_weight= np.concatenate([ewt, ewt]).astype(np.float32)

    data = Data(
        x=torch.from_numpy(x_np),
        edge_index=torch.from_numpy(edge_index),
        edge_type=torch.from_numpy(edge_type),
        edge_weight=torch.from_numpy(edge_weight),
        num_nodes=N,
    )
    data.issuer_index = torch.from_numpy(issuer_index)
    return data

def load_pyg_graph(base_dir: Path) -> Data:
    """
    base_dir is the features run folder that contains:
      - pyg_graph.pt   (preferred)
      - graph_nodes.(parquet|csv), graph_edges.(parquet|csv)  (fallback)
    """
    base_dir = Path(base_dir)
    pt = base_dir / "pyg_graph.pt"
    if pt.exists():
        try:
            # Torch 2.4+ may enforce safe loading that defaults to weights_only=True in some envs;
            # explicitly disable it because this file stores a full torch_geometric.data.Data object.
            return torch.load(pt, map_location="cpu", weights_only=False)  # type: ignore[arg-type]
        except Exception:
            pass
    # fallback from parquet/csv
    for nodes_name in ("graph_nodes.parquet", "graph_nodes.csv"):
        for edges_name in ("graph_edges.parquet", "graph_edges.csv"):
            np_nodes = base_dir / nodes_name
            np_edges = base_dir / edges_name
            if np_nodes.exists() and np_edges.exists():
                return _rebuild_data_from_tabular(np_nodes, np_edges)
    raise FileNotFoundError(f"No PyG graph or tabular edges/nodes found under {base_dir}")

# ---------------------------
# Basket/sample builder
# ---------------------------
@dataclass
class Sample:
    target_node: int
    port_nodes: List[int]
    port_weights: List[float]
    residual: float
    liq_ref: float
    dt_ord: int
    base_feats: List[float]
    ctx_feats: List[float]  # [log_sum_abs_w, signed_frac, frac_same_issuer]

def _extract_pf_base(x: str) -> Optional[str]:
    if not isinstance(x, str): return None
    m = re.match(r"^(PF_\d{8})", x)  # PF_YYYYMMDD_xxx -> PF_YYYYMMDD
    return m.group(1) if m else None

def _get_sign_col(df: pd.DataFrame) -> pd.Series:
    if "sign" in df and df["sign"].notna().any():
        s = pd.to_numeric(df["sign"], errors="coerce")
    elif "side" in df:
        # Convention: SELL = +1 (widening), BUY = -1 (tightening), matching simulator's side_sign
        s = df["side"].map({"CBUY": -1, "BUY": -1, "CSELL": 1, "SELL": 1})
    else:
        s = pd.Series([np.nan] * len(df))
    return s

def _ensure_trade_dt(df: pd.DataFrame) -> pd.Series:
    if "trade_dt" in df:
        return pd.to_datetime(df["trade_dt"], errors="coerce").dt.normalize()
    if "exec_time" in df:
        return pd.to_datetime(df["exec_time"], errors="coerce").dt.normalize()
    return pd.NaT

def _dv01_dollar(df: pd.DataFrame) -> pd.Series:
    if "dv01_dollar" in df and df["dv01_dollar"].notna().any():
        return pd.to_numeric(df["dv01_dollar"], errors="coerce")
    a = pd.to_numeric(df.get("dv01_per_100", np.nan), errors="coerce")
    q = pd.to_numeric(df.get("quantity_par", np.nan), errors="coerce")
    return a * q / 100.0

def _residual_label(row: pd.Series) -> float:
    for k in ["residual", "price_residual", "delta_residual"]:
        if k in row and pd.notna(row[k]):
            return float(row[k])
    # SIM data path
    if "y_bps" in row and "y_pi_ref_bps" in row and pd.notna(row["y_bps"]) and pd.notna(row["y_pi_ref_bps"]):
        return float(row["y_bps"]) - float(row["y_pi_ref_bps"])
    # Fallbacks used for real TRACE-style tables
    for a, b in [("exec_clean", "eval_clean"), ("exec_clean", "vendor_fair"), ("price", "vendor_fair")]:
        if a in row and b in row and pd.notna(row[a]) and pd.notna(row[b]):
            return float(row[a]) - float(row[b])
    return 0.0

def _liq_ref(row: pd.Series, bonds_row: pd.Series) -> float:
    for k in ["vendor_liq", "vendor_liq_score", "liq_score", "liq_vendor", "liq_ref"]:
        if k in row and pd.notna(row[k]): return float(row[k])
    for k in ["vendor_liq", "vendor_liq_score", "liq_score"]:
        if k in bonds_row and pd.notna(bonds_row[k]): return float(bonds_row[k])
    ao = bonds_row.get("amount_outstanding", np.nan)
    # minimal proxy if nothing else is present
    return float(0.0 if pd.isna(ao) else 1.0 / np.sqrt(max(1.0, float(ao))))

def build_samples(
    trades: pd.DataFrame,
    nodes: pd.DataFrame,
    sampler: SamplerConfig,
) -> List[Sample] | tuple[List[Sample], Dict[str, np.ndarray]]:
    rng = np.random.default_rng(sampler.seed)
    t = trades.copy()

    t["sign"] = _get_sign_col(t)
    t["trade_dt"] = _ensure_trade_dt(t)
    t["dv01_dollar"] = _dv01_dollar(t)
    t["portfolio_id"] = t.get("portfolio_id", pd.Series([np.nan] * len(t))).replace(["", "None", "nan", "NaN"], np.nan)
    if "exec_time" in t.columns:
        t["exec_time"] = pd.to_datetime(t["exec_time"], errors="coerce")

    # --- groups: exact (portfolio_id, trade_dt) only by default ---
    groups: List[pd.DataFrame] = []

    exact = t[t["portfolio_id"].notna()].groupby(["portfolio_id", "trade_dt"])
    for _, gg in exact:
        if len(gg) >= 2:
            # keep original ordering for reproducibility
            groups.append(gg.sort_values(["trade_dt", "isin"]))

    # Optional: allow synthetic coalescing when explicitly requested
    if sampler.coalesce_pf_base:
        base = t[t["portfolio_id"].notna()].copy()
        base["pf_base"] = base["portfolio_id"].map(_extract_pf_base)
        base = base[base["pf_base"].notna()]
        for (_, dt), gg in base.groupby(["pf_base", "trade_dt"]):
            if len(gg) < 2:
                continue
            idx = gg.index.to_numpy()
            rng.shuffle(idx)
            pos = 0
            while pos < len(idx):
                bs = int(rng.integers(sampler.chunk_min, sampler.chunk_max + 1))
                batch_idx = idx[pos : pos + bs]
                if len(batch_idx) >= 2:
                    groups.append(gg.loc[batch_idx])
                pos += bs

    # Fallback: if nothing else found, bucket by (customer_id, time)
    if len(groups) == 0 and "customer_id" in t.columns:
        tb = t.copy()
        tb["time_bucket"] = (tb.get("exec_time", tb["trade_dt"]).fillna(tb["trade_dt"]).dt.floor(f"{sampler.bucket_minutes}min"))
        for _, gg in tb.groupby(["customer_id", "trade_dt", "time_bucket"]):
            if len(gg) >= 2:
                groups.append(gg)

    node_map = dict(zip(nodes["isin"], nodes["node_id"]))
    bonds_idx = nodes.set_index("isin")

    samples: List[Sample] = []
    # Diagnostics accumulators for residuals sanity probes (raw samples)
    diag_residual: List[float] = []
    diag_sum_abs_w: List[float] = []
    diag_sum_signed_w: List[float] = []
    diag_frac_same_issuer: List[float] = []
    diag_log_size: List[float] = []
    diag_side: List[float] = []
    diag_vendor_liq: List[float] = []

    for gg in groups:
        gg = gg.dropna(subset=["isin", "sign", "dv01_dollar", "trade_dt"])
        if len(gg) < 2:
            continue
        # precompute absolute DV01 and sign
        gg = gg.copy()
        gg["_dv01_abs"] = _dv01_dollar(gg).abs()
        gg["_sign"] = _get_sign_col(gg).astype(float)
        group_dt_ord = pd.to_datetime(gg["trade_dt"]).max().to_pydatetime().toordinal()

        for i, row in gg.iterrows():
            tgt_isin = row["isin"]
            if tgt_isin not in node_map:
                continue
            tgt_node = int(node_map[tgt_isin])

            # portfolio context is others in the same basket
            ctx = gg.drop(index=i)
            if len(ctx) == 0:
                continue

            port_nodes: List[int] = []
            port_weights: List[float] = []

            if sampler.weight_mode == "raw_scaled":
                # previous behavior (less stable)
                w = (ctx["_sign"] * _dv01_dollar(ctx) / float(sampler.expo_scale)).clip(-sampler.clip_w, sampler.clip_w)
                port_weights = w.astype(float).tolist()
            else:
                # signed DV01 fractions (unit-sum abs within context)
                denom = float(ctx["_dv01_abs"].sum())
                if denom <= 0:
                    continue
                w_abs_frac = (ctx["_dv01_abs"] / denom).astype(float).to_numpy()
                w_signed = w_abs_frac * ctx["_sign"].astype(float).to_numpy()
                port_weights = w_signed.tolist()

            port_isins = ctx["isin"].tolist()
            port_nodes = [int(node_map[s]) for s in port_isins if s in node_map]
            if len(port_nodes) == 0:
                continue

            b_row = bonds_idx.loc[tgt_isin] if tgt_isin in bonds_idx.index else {}
            r = _residual_label(row)
            L = _liq_ref(row, b_row)

            # --- diagnostics on RAW context (pre-normalization) ---
            try:
                sum_abs_w_raw = float(ctx["_dv01_abs"].sum())
            except Exception:
                sum_abs_w_raw = float("nan")
            try:
                sum_signed_w_raw = float((ctx["_sign"] * _dv01_dollar(ctx)).sum())
            except Exception:
                sum_signed_w_raw = float("nan")
            # same-issuer fraction in basket
            frac_same_issuer = float("nan")
            try:
                issuer_col = "issuer_id" if "issuer_id" in bonds_idx.columns else ("issuer" if "issuer" in bonds_idx.columns else None)
                if issuer_col is not None:
                    tgt_issuer = bonds_idx.at[tgt_isin, issuer_col] if tgt_isin in bonds_idx.index else None
                    if tgt_issuer is not None and len(port_isins) > 0:
                        ctx_issuers = bonds_idx.reindex(port_isins)[issuer_col].tolist()
                        same = [1 for u in ctx_issuers if pd.notna(u) and u == tgt_issuer]
                        frac_same_issuer = float(len(same)) / float(len(port_isins))
            except Exception:
                pass
            # trade size (log of absolute dv01) and side
            try:
                dv01_abs = float(abs(row.get("dv01_dollar", np.nan)))
                log_size = float(np.log(dv01_abs + 1e-6)) if not np.isnan(dv01_abs) else float("nan")
            except Exception:
                log_size = float("nan")
            try:
                side_val = float(row.get("_sign", np.nan))
            except Exception:
                side_val = float("nan")

            diag_residual.append(float(r))
            diag_sum_abs_w.append(float(sum_abs_w_raw))
            diag_sum_signed_w.append(float(sum_signed_w_raw))
            diag_frac_same_issuer.append(float(frac_same_issuer))
            diag_log_size.append(float(log_size))
            diag_side.append(float(side_val))
            diag_vendor_liq.append(float(L))

            # per-trade baseline features: [log_size, side]
            # trade features: log size (fallbacks) and side sign
            size_raw = row.get("size", np.nan)
            if pd.isna(size_raw):
                size_raw = row.get("quantity_par", np.nan)
            if pd.isna(size_raw):
                size_raw = row.get("dv01_dollar", 0.0)
            log_size = float(np.log1p(abs(float(size_raw))))
            sgn = float(row["sign"]) if pd.notna(row["sign"]) else 0.0
            base_feats = [log_size, sgn]  # keep small & stable; urgency can be appended later

            sum_signed_w_raw = float((ctx["_sign"] * _dv01_dollar(ctx)).sum())
            log_sum_abs = float(np.log1p(max(sum_abs_w_raw, 0.0)))
            signed_frac = float(sum_signed_w_raw) / (float(sum_abs_w_raw) + 1e-6)
            # frac_same_issuer already computed as `frac_same_issuer`
            ctx_feats = [log_sum_abs, signed_frac, float(frac_same_issuer)]

            samples.append(Sample(tgt_node, port_nodes, port_weights, r, L, group_dt_ord, base_feats, ctx_feats))

    # Sanity: unique (target, day) pairs vs total samples
    try:
        n_pairs = len({(s.target_node, s.dt_ord) for s in samples})
        print("unique (target, day) pairs / samples:", n_pairs, "/", len(samples))
    except Exception:
        pass

    # residual stats and checks
    vals = np.array([s.residual for s in samples], float)
    print("residual stats (bps):", np.nanmean(vals), np.nanstd(vals), np.quantile(vals, [0.1, 0.5, 0.9]))
    assert np.nanstd(vals) > 1e-6, "Residual labels are constant (check label mapping)."

    # package diagnostics
    try:
        diags = {
            "residual": np.array(diag_residual, dtype=float),
            "sum_abs_w": np.array(diag_sum_abs_w, dtype=float),
            "sum_signed_w": np.array(diag_sum_signed_w, dtype=float),
            "frac_same_issuer": np.array(diag_frac_same_issuer, dtype=float),
            "log_size": np.array(diag_log_size, dtype=float),
            "side": np.array(diag_side, dtype=float),
            "vendor_liq": np.array(diag_vendor_liq, dtype=float),
        }
    except Exception:
        diags = None

    return (samples, diags) if diags is not None else samples

# ---------------------------
# Torch dataset / collate
# ---------------------------
class PortDataset(Dataset):
    def __init__(self, samples: List[Sample], idx: np.ndarray):
        self.samples = samples
        self.idx = idx.astype(int)
    def __len__(self) -> int: return len(self.idx)
    def __getitem__(self, i: int) -> Sample: return self.samples[self.idx[i]]

def port_collate(batch: List[Sample]) -> Dict[str, torch.Tensor]:
    B = len(batch)
    tgt = torch.tensor([b.target_node for b in batch], dtype=torch.long)
    sizes = [len(b.port_nodes) for b in batch]
    # Pre-normalization per-sample abs weight sum (portfolio scale)
    port_abs_sum = torch.tensor([float(np.sum(np.abs(b.port_weights))) if len(b.port_weights) > 0 else 0.0 for b in batch], dtype=torch.float32)
    if sum(sizes) > 0:
        port_index = torch.tensor([n for b in batch for n in b.port_nodes], dtype=torch.long)
        port_weight = torch.tensor([w for b in batch for w in b.port_weights], dtype=torch.float32)
        port_batch = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(sizes)])
        # Normalize port weights per portfolio batch by absolute sum
        if port_weight is not None and port_weight.numel() > 0:
            abs_sum = torch.zeros(B, device=port_weight.device).scatter_add_(0, port_batch, port_weight.abs())
            scale = abs_sum.gather(0, port_batch) + 1e-6
            port_weight = port_weight / scale
    else:
        port_index = torch.empty(0, dtype=torch.long)
        port_weight = torch.empty(0, dtype=torch.float32)
        port_batch = torch.empty(0, dtype=torch.long)
    residual = torch.tensor([b.residual for b in batch], dtype=torch.float32)
    liq_ref  = torch.tensor([b.liq_ref  for b in batch], dtype=torch.float32)
    # Baseline per-trade features: tolerate legacy samples without base_feats
    try:
        if all(hasattr(b, "base_feats") and b.base_feats is not None for b in batch):
            baseline_feats = torch.tensor([b.base_feats for b in batch], dtype=torch.float32)
        else:
            baseline_feats = torch.zeros((B, 2), dtype=torch.float32)
    except Exception:
        baseline_feats = torch.zeros((B, 2), dtype=torch.float32)

    # Context scalars: tolerate legacy samples without ctx_feats
    try:
        ctx_feats_list = [b.ctx_feats if hasattr(b, "ctx_feats") and b.ctx_feats is not None else [0.0, 0.0, 0.0] for b in batch]
    except Exception:
        ctx_feats_list = [[0.0, 0.0, 0.0] for _ in batch]
    ctx_feats = torch.tensor(ctx_feats_list, dtype=torch.float32)

    return dict(target_index=tgt,
                port_index=port_index,
                port_batch=port_batch,
                port_weight=port_weight,
                residual=residual,
                liq_ref=liq_ref,
                baseline_feats=baseline_feats,
                sizes=sizes,
                port_abs_sum=port_abs_sum,
                ctx_feats=ctx_feats)

# ---------------------------
# Evaluation metrics (VAL)
# ---------------------------
@torch.no_grad()
def eval_epoch(model: LiquidityModelGAT, data: Data, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    preds_m, preds50, preds90, ys = [], [], [], []
    for batch in loader:
        tgt  = batch["target_index"].to(device)
        pidx = batch["port_index"].to(device)
        pbat = batch["port_batch"].to(device)
        pw   = batch["port_weight"].to(device) if batch["port_weight"].numel() > 0 else None
        r    = batch["residual"].to(device)

        bf   = batch.get("baseline_feats", torch.empty(0)).to(device) if isinstance(batch, dict) else None

        out = model.forward_from_data(
            data,
            tgt,
            pidx,
            pbat,
            pw,
            baseline_feats=bf,
            ctx_feats=batch["ctx_feats"].to(device)
        )

        preds_m.append(out["delta_mean"].flatten())
        preds50.append(out["q50"].flatten())
        preds90.append(out["q90"].flatten())
        ys.append(r)

    if not ys:  # dataset may be empty in edge tests
        return dict(mae=float("nan"), cov50=float("nan"), cov90=float("nan"),
                    width_mean=float("nan"), width_std=float("nan"))

    y = torch.cat(ys)
    m = torch.cat(preds_m); q5 = torch.cat(preds50); q9 = torch.cat(preds90)
    err = (m - y)
    mae = err.abs().mean().item()
    rmse = float(torch.sqrt((err * err).mean()).item())
    cov50 = (y <= q5).float().mean().item()
    cov90 = (y <= q9).float().mean().item()
    w = (q9 - q5)
    return dict(mae=mae, rmse=rmse, cov50=cov50, cov90=cov90, width_mean=w.mean().item(), width_std=w.std(unbiased=False).item())

# ---------------------------
# Label normalization helper (exported for tests)
# ---------------------------

def compute_label_norm(y_train: np.ndarray | list[float]) -> tuple[float, float]:
    """Compute (mean, scale) for label normalization matching the training loop.
    Scale uses MAD*1.4826 with std fallback if MAD is degenerate.
    Returns (mean_bps, scale_bps).
    """
    y_arr = np.asarray(y_train, dtype=float)
    y_mean = float(np.nanmean(y_arr)) if y_arr.size > 0 else 0.0
    med = float(np.nanmedian(y_arr)) if y_arr.size > 0 else 0.0
    mad = float(np.nanmedian(np.abs(y_arr - med))) * 1.4826 if y_arr.size > 0 else 0.0
    y_std = float(mad if mad > 1e-6 else (np.nanstd(y_arr) + 1e-6))
    return y_mean, y_std

# ---------------------------
# Scheduling helpers (exported for tests)
# ---------------------------

def sched_alpha(epoch: int) -> float:
    """Deterministic schedule for alpha (weight of q50) by epoch (0-based).
    Epochs 0–2: 0.0; 3–5: linear 0.1→0.6; ≥6: 1.0
    """
    if epoch <= 2:
        return 0.0
    if epoch <= 5:
        frac = (epoch - 3) / 2.0
        frac = float(max(0.0, min(1.0, frac)))
        return 0.1 + (0.6 - 0.1) * frac
    return 1.0

def sched_lambda_huber(epoch: int) -> float:
    """Deterministic schedule for lambda_huber by epoch (0-based).
    Epochs 0–2: 1.0; 3–5: linear 1.0→0.3; ≥6: 0.2
    """
    if epoch <= 2:
        return 1.0
    if epoch <= 5:
        frac = (epoch - 3) / 2.0
        frac = float(max(0.0, min(1.0, frac)))
        return 1.0 + (0.3 - 1.0) * frac
    return 0.2

def sched_lambda_noncross(epoch: int) -> float:
    """Monotonicity guardrail (non-crossing) — stays on at 0.1 in tests."""
    return 0.10

def sched_lambda_wmono(epoch: int) -> float:
    """Width monotonicity — 0.0 until epoch ≥6, then 0.10."""
    return 0.0 if epoch <= 5 else 0.10

def sched_beta_q90(epoch: int) -> float:
    """Quantile-90 weight — kept 0.0 in tests until explicitly enabled later."""
    return 0.0

# ---------------------------
# Orchestrator
# ---------------------------

def train_gat(
    features_run_dir: Path,
    trades_path: Path,
    outdir: Path,
    run_cfg: LiquidityRunConfig,
    ranges_json: Optional[Path] = None,
    graph_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    dev = _device(run_cfg.train.device)
    torch.manual_seed(run_cfg.train.seed)
    np.random.seed(run_cfg.train.seed)

    # graph + nodes
    data = load_pyg_graph(Path(features_run_dir))
    nodes_tab = None
    # Determine where to read nodes/edges from: prefer explicit graph_dir, fallback to features_run_dir
    base_nodes_dir = Path(graph_dir) if graph_dir is not None else Path(features_run_dir)
    for name in ("graph_nodes.parquet", "graph_nodes.csv"):
        p = base_nodes_dir / name
        if p.exists():
            nodes_tab = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
            break
    if nodes_tab is None:
        raise FileNotFoundError(f"Could not find graph_nodes.(parquet|csv) in {base_nodes_dir}")

    # trades
    trades = pd.read_parquet(trades_path) if str(trades_path).endswith(".parquet") else pd.read_csv(trades_path)

    # build samples
    samples_out = build_samples(trades, nodes_tab, run_cfg.sampler)
    diags = None
    if isinstance(samples_out, tuple):
        samples, diags = samples_out  # type: ignore[assignment]
    else:
        samples = samples_out  # type: ignore[assignment]

    if not samples:
        raise RuntimeError("No training samples built. Check grouping and columns in trades.")

    # Residuals sanity probes on RAW samples (printed now; TB logging after writer is created)
    residual_corr: Dict[str, float] = {}
    if isinstance(diags, dict) and "residual" in diags:
        def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
            try:
                x = np.asarray(x, dtype=float)
                y = np.asarray(y, dtype=float)
                m = np.isfinite(x) & np.isfinite(y)
                x = x[m]; y = y[m]
                if x.size < 3:
                    return float("nan")
                sx = float(np.nanstd(x)); sy = float(np.nanstd(y))
                if sx <= 1e-12 or sy <= 1e-12:
                    return float("nan")
                return float(np.corrcoef(x, y)[0, 1])
            except Exception:
                return float("nan")
        y = diags.get("residual")
        metrics = {
            "sum_abs_w": diags.get("sum_abs_w"),
            "sum_signed_w": diags.get("sum_signed_w"),
            "frac_same_issuer": diags.get("frac_same_issuer"),
            "log_size": diags.get("log_size"),
            "side": diags.get("side"),
            "vendor_liq": diags.get("vendor_liq"),
        }
        print("Residuals sanity probes (Pearson r):")
        for k, x in metrics.items():
            if x is None:
                continue
            r = safe_corr(y, x)
            residual_corr[k] = r
            print(f"  corr(residual, {k}) = {r:.4f}")

    dates = np.array([s.dt_ord for s in samples], dtype=int)
    if ranges_json and Path(ranges_json).exists():
        spans = json.loads(Path(ranges_json).read_text())
        # derive masks against trade_dt (ordinal)
        def _to_mask(span: Dict[str, str]):
            s = pd.to_datetime(span["start"]).to_pydatetime().toordinal()
            e = pd.to_datetime(span["end"]).to_pydatetime().toordinal()
            return (dates >= s) & (dates <= e)
        mtr = _to_mask(spans["train"]); mva = _to_mask(spans["val"])
        idx_train = np.where(mtr)[0]; idx_val = np.where(mva)[0]
    else:
        cut = np.quantile(dates, 0.8)
        idx_train = np.where(dates <= cut)[0]
        idx_val   = np.where(dates >  cut)[0]
        if len(idx_val) == 0:
            # fallback 80/20 on samples
            perm = np.random.default_rng(7).permutation(len(samples))
            cutn = int(0.8 * len(samples))
            idx_train, idx_val = perm[:cutn], perm[cutn:]

    train_ds = PortDataset(samples, idx_train)
    val_ds   = PortDataset(samples, idx_val)
    train_loader = DataLoader(train_ds, batch_size=int(run_cfg.train.batch_size), shuffle=True,  collate_fn=port_collate, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=int(run_cfg.train.batch_size), shuffle=False, collate_fn=port_collate, drop_last=False)

    # --- label normalization from train set ---
    y_train = np.array([samples[i].residual for i in idx_train], dtype=float)
    y_mean  = float(np.nanmean(y_train))
    mad = float(np.nanmedian(np.abs(y_train - np.nanmedian(y_train)))) * 1.4826
    y_std  = float(mad if mad > 1e-6 else np.nanstd(y_train) + 1e-6)
    print(f"[label norm] mean={y_mean:.3f} bps, scale={y_std:.3f} bps")

    def _znorm_y(t: torch.Tensor) -> torch.Tensor:
        return (t - y_mean) / y_std

    def _znorm_pred(pred: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            'delta_mean': (pred['delta_mean'] - y_mean) / y_std,
            'q50':        (pred['q50']        - y_mean) / y_std,
            'q90':        (pred['q90']        - y_mean) / y_std,
        }

    # --- simple baseline: OLS on [const, sign, log_size] from train split ---
    try:
        Xb, yb = [], []
        for i in idx_train:
            log_size, sgn = samples[i].base_feats  # [log_size, sign]
            Xb.append([1.0, float(sgn), float(log_size)])
            yb.append(float(samples[i].residual))
        w_ls, *_ = np.linalg.lstsq(np.asarray(Xb), np.asarray(yb), rcond=None)
        print(f"[baseline OLS] w={w_ls}")
        # Precompute baseline MAE on validation split
        Xv, yv = [], []
        for i in idx_val:
            log_size, sgn = samples[i].base_feats
            Xv.append([1.0, float(sgn), float(log_size)])
            yv.append(float(samples[i].residual))
        Xv = np.asarray(Xv); yv = np.asarray(yv)
        yv_hat = Xv @ w_ls
        baseline_val_mae = float(np.mean(np.abs(yv_hat - yv))) if yv.size > 0 else float("nan")
    except Exception as e:
        print(f"[baseline OLS] failed: {e}")
        w_ls = None
        baseline_val_mae = float("nan")

    # model
    num_rel = int(data.edge_type.max().item() + 1)
    model = LiquidityModelGAT(
        x_dim=int(data.x.size(1)),
        num_relations=num_rel,
        d_model=run_cfg.model.d_model,
        issuer_emb_dim=run_cfg.model.issuer_emb_dim,
        dropout=run_cfg.model.dropout,
        heads=run_cfg.model.heads,
        rel_emb_dim=run_cfg.model.rel_emb_dim,
        rel_init_boost=run_cfg.model.rel_init_boost or {},
        encoder_type=getattr(run_cfg.model, 'encoder_type', 'gat'),
        baseline_dim=2,
    ).to(dev)
    data = data.to(dev)  # keep once on device

    # opt/sched
    opt = torch.optim.AdamW(model.parameters(), lr=_safe_float(run_cfg.train.lr, 1e-3),
                            weight_decay=_safe_float(run_cfg.train.weight_decay, 1e-4))
    # Learning-rate warmup + cosine via a single LambdaLR stepped per-iteration
    try:
        steps_per_epoch = max(1, len(train_loader))
        total_steps = max(steps_per_epoch * int(run_cfg.train.max_epochs), 1)
        warmup_steps = min(100, steps_per_epoch * 2)
        start_factor = 0.2
        cosine_steps = max(total_steps - warmup_steps, 1)
        def lr_lambda(step: int) -> float:
            # step is the global optimizer step index (0-based)
            if step < warmup_steps:
                # linear warmup from start_factor to 1.0
                return float(start_factor + (1.0 - start_factor) * (step / max(1, warmup_steps)))
            t = (step - warmup_steps) / float(cosine_steps)
            t = min(max(t, 0.0), 1.0)
            return float(0.5 * (1.0 + math.cos(math.pi * t)))
        sched = LambdaLR(opt, lr_lambda=lr_lambda)
    except Exception:
        sched = None  # scheduler optional
    # EMA of weights for eval
    ema = ModelEMA(model, decay=0.995).to(dev)

    best = float("inf"); best_rmse = float("inf"); best_ep = -1; patience_left = int(run_cfg.train.patience)

    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "progress.jsonl").touch()

    # TensorBoard writer (optional)
    writer = None
    if getattr(run_cfg.train, "enable_tb", True) and SummaryWriter is not None:
        tb_dir = Path(run_cfg.train.tb_log_dir) if run_cfg.train.tb_log_dir else (outdir / "tb")
        try:
            tb_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=str(tb_dir))
            # dump config for reference
            try:
                writer.add_text("config/train", json.dumps(asdict(run_cfg.train), indent=2).replace("\n", "  \n"), 0)
                writer.add_text("config/model", json.dumps(asdict(run_cfg.model), indent=2).replace("\n", "  \n"), 0)
                writer.add_text("config/sampler", json.dumps(asdict(run_cfg.sampler), indent=2).replace("\n", "  \n"), 0)
            except Exception:
                pass
            # Log residuals sanity probes if computed
            try:
                if 'residual_corr' in locals() and isinstance(residual_corr, dict) and len(residual_corr) > 0:
                    for k, v in residual_corr.items():
                        writer.add_scalar(f"residuals/{k}_corr", float(v), 0)
            except Exception:
                pass
            print(f"TensorBoard logging to: {tb_dir}  |  run: tensorboard --logdir {tb_dir.parent}")
        except Exception:
            writer = None

    def beta_schedule(epoch: int, warmup: int = 2, target: float = 0.6, ramp: int = 8) -> float:
        """Gentle monotone ramp for beta: 0 for first `warmup` epochs, then linear to `target` over `ramp` epochs."""
        if epoch <= warmup:
            return 0.0
        e = epoch - warmup
        if e >= ramp:
            return target
        return target * (e / ramp)

    def loss_weight_schedule(epoch: int) -> Dict[str, float]:
        # Warm up on Huber, then ramp q50 up while dialing Huber down.
        # Keep non-crossing on throughout; delay width-monotonicity until epoch 6.
        if epoch <= 2:
            return dict(alpha=0.0, lambda_huber=1.0, lambda_noncross=0.10, lambda_wmono=0.0)
        elif epoch <= 5:
            # Epochs 3,4,5: alpha ramps 0.1 -> 0.6 linearly; lambda_huber 1.0 -> 0.3 linearly
            # When called from the main loop (ep starts at 1), map ep=3..5 to frac=0..1
            frac = (epoch - 3) / 2.0
            frac = float(max(0.0, min(1.0, frac)))
            alpha = 0.1 + (0.6 - 0.1) * frac
            lambda_huber = 1.0 + (0.3 - 1.0) * frac
            return dict(alpha=alpha, lambda_huber=lambda_huber, lambda_noncross=0.10, lambda_wmono=0.0)
        else:
            return dict(alpha=1.0, lambda_huber=0.2, lambda_noncross=0.10, lambda_wmono=0.10)

    # train
    global_step = 0
    for ep in range(1, int(run_cfg.train.max_epochs) + 1):
        model.train()
        ep_loss = ep_mae = ep_mse = 0.0; nobs = 0
        # Epoch accumulators for training-stream diagnostics
        ep_y = []         # residual labels
        ep_m = []         # delta_mean predictions
        ep_q50 = []
        ep_q90 = []
        ep_port_abs_sum = 0.0

        iterable = train_loader
        if tqdm is not None:
            try:
                iterable = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {ep}", leave=False)
            except Exception:
                iterable = train_loader

        for step, batch in enumerate(iterable, 1):
            tgt  = batch["target_index"].to(dev)
            pidx = batch["port_index"].to(dev)
            pbat = batch["port_batch"].to(dev)
            pw   = batch["port_weight"].to(dev) if batch["port_weight"].numel() > 0 else None
            r    = batch["residual"].to(dev)
            Lr   = batch["liq_ref"].to(dev)

            bf   = batch.get("baseline_feats", torch.empty(0)).to(dev) if isinstance(batch, dict) else None
            out = model.forward_from_data(data, tgt, pidx, pbat, pw, baseline_feats=bf, ctx_feats=batch["ctx_feats"].to(dev))
            # schedule the weights: emphasis on Huber early, then q50; delay q90
            wts = loss_weight_schedule(ep)
            # normalize labels and predictions only inside the loss
            pred_n = _znorm_pred(out)
            r_n    = _znorm_y(r)
            losses = composite_loss(
                pred_n, r_n, Lr,
                alpha=wts['alpha'],
                beta=(0.0 if ep <= 10 else 0.2),  # start quantiles late and small
                gamma=0.1,
                delta_huber=1.0,                  # 1.0 now in normalized units
                lambda_huber=wts['lambda_huber'],
                lambda_noncross=wts['lambda_noncross'],
                lambda_wmono=wts['lambda_wmono'],
                lambda_wsize=0.005
            )

            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            # gradient/global norm (pre-clipping) returned by clip_grad_norm_
            grad_norm = clip_grad_norm_(model.parameters(), float(run_cfg.train.clip_grad))
            opt.step()
            # scheduler + EMA updates
            try:
                if sched is not None:
                    sched.step()
            except Exception:
                pass
            try:
                ema.update(model)
            except Exception:
                pass

            B = int(tgt.numel())
            with torch.no_grad():
                dm = out["delta_mean"].flatten()
                q50 = out["q50"].flatten()
                q90 = out["q90"].flatten()
                w = (q90 - q50)
                err = (dm - r)
                mae = err.abs().mean()
                mse = (err * err).mean()
                # batch quantile diagnostics
                q50_min, q50_max, q50_mean = q50.min(), q50.max(), q50.mean()
                q90_min, q90_max, q90_mean = q90.min(), q90.max(), q90.mean()
                w_min, w_max, w_mean, w_std = w.min(), w.max(), w.mean(), w.std(unbiased=False)
                cov50_b = (r <= q50).float().mean()
                cov90_b = (r <= q90).float().mean()
                # portfolio scale
                batch_port_abs_sum = float(batch.get("port_abs_sum", torch.tensor([])).sum().item()) if "port_abs_sum" in batch else 0.0
                basket_sizes = batch.get("sizes", [])
                basket_mean = float(np.mean(basket_sizes)) if len(basket_sizes) > 0 else 0.0

            # accumulate epoch-level arrays
            ep_y.append(r.detach().cpu())
            ep_m.append(dm.detach().cpu())
            ep_q50.append(q50.detach().cpu())
            ep_q90.append(q90.detach().cpu())
            ep_port_abs_sum += batch_port_abs_sum

            ep_loss += float(losses["total"].item()) * B
            ep_mae  += float(mae.item()) * B
            ep_mse  += float(mse.item()) * B
            nobs    += B

            # TensorBoard step logging
            if writer is not None:
                try:
                    writer.add_scalar("train/loss", float(losses["total"].item()), global_step)
                    writer.add_scalar("train/mae", float(mae.item()), global_step)
                    writer.add_scalar("train/rmse", float(torch.sqrt(mse).item()), global_step)
                    # Mean–median gap (diagnostic)
                    writer.add_scalar("train/mean_median_gap", float((dm - q50).abs().mean().item()), global_step)
                    # Quantile diagnostics per batch
                    writer.add_scalar("train/q50_min", float(q50_min.item()), global_step)
                    writer.add_scalar("train/q50_max", float(q50_max.item()), global_step)
                    writer.add_scalar("train/q50_mean", float(q50_mean.item()), global_step)
                    writer.add_scalar("train/q90_min", float(q90_min.item()), global_step)
                    writer.add_scalar("train/q90_max", float(q90_max.item()), global_step)
                    writer.add_scalar("train/q90_mean", float(q90_mean.item()), global_step)
                    writer.add_scalar("train/width_min", float(w_min.item()), global_step)
                    writer.add_scalar("train/width_max", float(w_max.item()), global_step)
                    writer.add_scalar("train/width_mean", float(w_mean.item()), global_step)
                    writer.add_scalar("train/width_std", float(w_std.item()), global_step)
                    # Coverage (training batches)
                    writer.add_scalar("train/cov50", float(cov50_b.item()), global_step)
                    writer.add_scalar("train/cov90", float(cov90_b.item()), global_step)
                    # Gradient/global norm and portfolio scale
                    writer.add_scalar("train/grad_norm", float(grad_norm.item()) if hasattr(grad_norm, 'item') else float(grad_norm), global_step)
                    writer.add_scalar("train/port_absw_sum", float(batch_port_abs_sum), global_step)
                    writer.add_scalar("train/basket_size_mean", float(basket_mean), global_step)
                except Exception:
                    pass
            global_step += 1

            if (step % int(run_cfg.train.print_every)) == 0:
                msg = f"[ep {ep} step {step}] batch={B} loss={losses['total'].item():.4f} huber={losses['huber'].item():.4f} " \
                      f"mae={mae.item():.4f} wpen={losses.get('wpen', torch.tensor(0.0)).item():.4f}"
                if tqdm is not None and hasattr(iterable, 'set_postfix'):
                    try:
                        iterable.set_postfix(loss=f"{losses['total'].item():.4f}", mae=f"{mae.item():.4f}")
                    except Exception:
                        pass
                else:
                    print(msg)

        # end epoch: eval
        train_mae = ep_mae / max(1, nobs)
        train_rmse = float(np.sqrt(ep_mse / max(1, nobs)))
        # Training-stream diagnostics
        with torch.no_grad():
            y_tr = torch.cat(ep_y) if len(ep_y) else torch.tensor([])
            m_tr = torch.cat(ep_m) if len(ep_m) else torch.tensor([])
            q50_tr = torch.cat(ep_q50) if len(ep_q50) else torch.tensor([])
            q90_tr = torch.cat(ep_q90) if len(ep_q90) else torch.tensor([])
            width_tr = (q90_tr - q50_tr) if q50_tr.numel() > 0 else torch.tensor([])
            if y_tr.numel() > 0:
                cov50_tr = float((y_tr <= q50_tr).float().mean().item())
                cov90_tr = float((y_tr <= q90_tr).float().mean().item())
                width_mean_tr = float(width_tr.mean().item())
                width_std_tr  = float(width_tr.std(unbiased=False).item())
                # label stats
                y_np = y_tr.cpu().numpy()
                label_mean = float(y_np.mean())
                label_std  = float(y_np.std(ddof=0))
                label_p10  = float(np.quantile(y_np, 0.10))
                label_p50  = float(np.quantile(y_np, 0.50))
                label_p90  = float(np.quantile(y_np, 0.90))
            else:
                cov50_tr = cov90_tr = width_mean_tr = width_std_tr = float("nan")
                label_mean = label_std = label_p10 = label_p50 = label_p90 = float("nan")
        # Parameter norm (L2)
        with torch.no_grad():
            try:
                param_sq_sum = 0.0
                for p in model.parameters():
                    if p.requires_grad:
                        param_sq_sum += float(p.detach().pow(2).sum().item())
                param_norm = float(np.sqrt(param_sq_sum))
            except Exception:
                param_norm = float("nan")
        # Attention sanity: exp(rel_log_gain) by relation
        attn_rel = []
        try:
            gains = torch.exp(model.encoder.rel_log_gain.detach().cpu())
            attn_rel = gains.tolist()
        except Exception:
            attn_rel = []

        # Evaluate with EMA-smoothed weights for stability
        val_metrics = eval_epoch(ema.ema if ema is not None else model, data, val_loader, dev)
        val_mae = float(val_metrics["mae"])
        val_rmse = float(val_metrics.get("rmse", float("nan")))

        # log (file)
        with (outdir / "progress.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "epoch": ep,
                "train_mae": train_mae,
                "train_rmse": train_rmse,
                "baseline_mae": baseline_val_mae,
                "train": {
                    "cov50": cov50_tr, "cov90": cov90_tr,
                    "width_mean": width_mean_tr, "width_std": width_std_tr,
                    "label_mean": label_mean, "label_std": label_std,
                    "label_p10": label_p10, "label_p50": label_p50, "label_p90": label_p90,
                    "port_absw_sum": ep_port_abs_sum,
                },
                "val": val_metrics
            }) + "\n")

        # TensorBoard epoch logging
        if writer is not None:
            try:
                # Training-stream metrics
                writer.add_scalar("epoch/train_mae", float(train_mae), ep)
                writer.add_scalar("epoch/train_rmse", float(train_rmse), ep)
                writer.add_scalar("epoch/train_cov50", float(cov50_tr), ep)
                writer.add_scalar("epoch/train_cov90", float(cov90_tr), ep)
                writer.add_scalar("epoch/train_width_mean", float(width_mean_tr), ep)
                writer.add_scalar("epoch/train_width_std", float(width_std_tr), ep)
                writer.add_scalar("epoch/train_label_mean", float(label_mean), ep)
                writer.add_scalar("epoch/train_label_std", float(label_std), ep)
                writer.add_scalar("epoch/train_label_p10", float(label_p10), ep)
                writer.add_scalar("epoch/train_label_p50", float(label_p50), ep)
                writer.add_scalar("epoch/train_label_p90", float(label_p90), ep)
                writer.add_scalar("epoch/train_port_absw_sum", float(ep_port_abs_sum), ep)
                writer.add_scalar("epoch/train_param_norm", float(param_norm), ep)
                # Attention gains by relation
                for i, g in enumerate(attn_rel):
                    writer.add_scalar(f"attn/rel_gain_exp/{i}", float(g), ep)
                # Validation metrics
                writer.add_scalar("epoch/val_mae", float(val_mae), ep)
                writer.add_scalar("epoch/val_rmse", float(val_rmse), ep)
                writer.add_scalar("epoch/val_cov50", float(val_metrics["cov50"]), ep)
                writer.add_scalar("epoch/val_cov90", float(val_metrics["cov90"]), ep)
                writer.add_scalar("epoch/val_width_mean", float(val_metrics["width_mean"]), ep)
                writer.add_scalar("epoch/val_width_std", float(val_metrics["width_std"]), ep)
                # Baseline (vendor-only) MAE for comparison
                try:
                    writer.add_scalar("epoch/baseline_mae", float(baseline_val_mae), ep)
                except Exception:
                    pass
                # learning rate (first param group)
                try:
                    lr0 = float(opt.param_groups[0].get("lr", float("nan")))
                    writer.add_scalar("epoch/lr", lr0, ep)
                except Exception:
                    pass
            except Exception:
                pass

        print(f"Epoch {ep:02d} | train_mae={train_mae:.4f} | train_rmse={train_rmse:.4f} | val_mae={val_mae:.4f} | val_rmse={val_rmse:.4f} | cov50={val_metrics['cov50']:.3f} "
              f"| cov90={val_metrics['cov90']:.3f} | width={val_metrics['width_mean']:.3f}±{val_metrics['width_std']:.3f}")

        # early stop
        if val_mae + 1e-8 < best:
            best = val_mae; best_rmse = val_rmse; best_ep = ep; patience_left = int(run_cfg.train.patience)
            ckpt = {
                "arch": {"type": "liquidity_model_gat",
                         "x_dim": int(data.x.size(1)),
                         "num_relations": num_rel,
                         "model": asdict(run_cfg.model)},
                "state_dict": model.state_dict(),
                "ema_state_dict": ema.ema.state_dict(),
                "best": {"epoch": best_ep, "val_mae": float(best), "val_rmse": float(val_rmse)},
                "train_config": asdict(run_cfg.train),
            }
            torch.save(ckpt, outdir / "ckpt_liquidity.pt")
            torch.save(ckpt, outdir / "ckpt.pt")
            (outdir / "metrics_val.json").write_text(json.dumps({"best_epoch": best_ep, "best_val_mae_bps": float(best), "best_val_rmse_bps": float(val_rmse)}, indent=2))
            # lightweight calibration snapshot
            with open(outdir / "calibration.json", "w", encoding="utf-8") as jf:
                jf.write(json.dumps({
                    "epoch": best_ep,
                    "cov50": val_metrics["cov50"],
                    "cov90": val_metrics["cov90"],
                    "width_mean": val_metrics["width_mean"],
                    "width_std": val_metrics["width_std"],
                }, indent=2))
            print("  ✓ Saved best checkpoint and calibration JSON.")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {ep}.")
                break

    # finalize TB
    if writer is not None:
        try:
            writer.flush()
            writer.close()
        except Exception:
            pass
    return {"best_epoch": best_ep, "best_val_mae_bps": float(best), "best_val_rmse_bps": float(best_rmse)}

# Backward-compatible alias if other modules still import the old name
train_liquidity = train_gat
