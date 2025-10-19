# ptliq/training/gat_loop.py
from __future__ import annotations

import json, re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import Data

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
            return torch.load(pt, map_location="cpu")
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

def _extract_pf_base(x: str) -> Optional[str]:
    if not isinstance(x, str): return None
    m = re.match(r"^(PF_\d{8})", x)  # PF_YYYYMMDD_xxx -> PF_YYYYMMDD
    return m.group(1) if m else None

def _get_sign_col(df: pd.DataFrame) -> pd.Series:
    if "sign" in df and df["sign"].notna().any():
        s = pd.to_numeric(df["sign"], errors="coerce")
    elif "side" in df:
        s = df["side"].map({"CBUY": 1, "BUY": 1, "CSELL": -1, "SELL": -1})
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
        if k in row and pd.notna(row[k]): return float(row[k])
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
) -> List[Sample]:
    rng = np.random.default_rng(sampler.seed)
    t = trades.copy()

    t["sign"] = _get_sign_col(t)
    t["trade_dt"] = _ensure_trade_dt(t)
    t["dv01_dollar"] = _dv01_dollar(t)
    t["portfolio_id"] = t.get("portfolio_id", pd.Series([np.nan] * len(t))).replace(["", "None", "nan", "NaN"], np.nan)
    if "exec_time" in t.columns:
        t["exec_time"] = pd.to_datetime(t["exec_time"], errors="coerce")

    base = t[t["portfolio_id"].notna()].copy()
    base["pf_base"] = base["portfolio_id"].map(_extract_pf_base)

    groups: List[pd.DataFrame] = []
    for (_pfb, dt), gg in base.groupby([base["pf_base"], base["trade_dt"]]):
        if len(gg) < 2:
            continue
        idx = gg.index.to_numpy()
        rng.shuffle(idx)
        pos = 0
        while pos < len(idx):
            bs = int(rng.integers(sampler.chunk_min, sampler.chunk_max + 1))
            batch_idx = idx[pos:pos + bs]
            if len(batch_idx) >= 2:
                groups.append(gg.loc[batch_idx])
            pos += bs

    if len(groups) == 0 and "customer_id" in t.columns:
        tb = t.copy()
        tb["time_bucket"] = (tb.get("exec_time", tb["trade_dt"]).fillna(tb["trade_dt"]).dt.floor(f"{sampler.bucket_minutes}min"))
        for _, gg in tb.groupby(["customer_id", "trade_dt", "time_bucket"]):
            if len(gg) >= 2:
                groups.append(gg)

    node_map = dict(zip(nodes["isin"], nodes["node_id"]))
    bonds_idx = nodes.set_index("isin")

    samples: List[Sample] = []
    for gg in groups:
        gg = gg.dropna(subset=["isin", "sign", "dv01_dollar", "trade_dt"])
        if len(gg) < 2:
            continue
        w = (gg["sign"] * gg["dv01_dollar"] / float(sampler.expo_scale)).clip(-sampler.clip_w, sampler.clip_w)
        gg = gg.assign(_w=w.values)
        group_dt_ord = pd.to_datetime(gg["trade_dt"]).max().to_pydatetime().toordinal()

        for i, row in gg.iterrows():
            tgt_isin = row["isin"]
            if tgt_isin not in node_map:
                continue
            tgt_node = int(node_map[tgt_isin])
            ctx = gg.drop(index=i)
            port_isins = ctx["isin"].tolist()
            port_nodes = [int(node_map[s]) for s in port_isins if s in node_map]
            if len(port_nodes) == 0:
                continue
            port_weights = ctx["_w"].astype(float).tolist()
            b_row = bonds_idx.loc[tgt_isin] if tgt_isin in bonds_idx.index else {}
            r = _residual_label(row)
            L = _liq_ref(row, b_row)
            samples.append(Sample(tgt_node, port_nodes, port_weights, r, L, group_dt_ord))
    return samples

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
    if sum(sizes) > 0:
        port_index = torch.tensor([n for b in batch for n in b.port_nodes], dtype=torch.long)
        port_weight= torch.tensor([w for b in batch for w in b.port_weights], dtype=torch.float32)
        port_batch = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(sizes)])
    else:
        port_index = torch.empty(0, dtype=torch.long)
        port_weight= torch.empty(0, dtype=torch.float32)
        port_batch = torch.empty(0, dtype=torch.long)
    residual = torch.tensor([b.residual for b in batch], dtype=torch.float32)
    liq_ref  = torch.tensor([b.liq_ref  for b in batch], dtype=torch.float32)
    return dict(target_index=tgt, port_index=port_index, port_batch=port_batch,
                port_weight=port_weight, residual=residual, liq_ref=liq_ref, sizes=sizes)

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

        out = model.forward_from_data(data, tgt, pidx, pbat, pw)
        preds_m.append(out["delta_mean"].flatten())
        preds50.append(out["q50"].flatten())
        preds90.append(out["q90"].flatten())
        ys.append(r)

    if not ys:  # dataset may be empty in edge tests
        return dict(mae=float("nan"), cov50=float("nan"), cov90=float("nan"),
                    width_mean=float("nan"), width_std=float("nan"))

    y = torch.cat(ys)
    m = torch.cat(preds_m); q5 = torch.cat(preds50); q9 = torch.cat(preds90)
    mae = (m - y).abs().mean().item()
    cov50 = (y <= q5).float().mean().item()
    cov90 = (y <= q9).float().mean().item()
    w = (q9 - q5)
    return dict(mae=mae, cov50=cov50, cov90=cov90, width_mean=w.mean().item(), width_std=w.std(unbiased=False).item())

# ---------------------------
# Orchestrator
# ---------------------------
def train_gat(
    features_run_dir: Path,
    trades_path: Path,
    outdir: Path,
    run_cfg: LiquidityRunConfig,
    ranges_json: Optional[Path] = None,
) -> Dict[str, Any]:
    dev = _device(run_cfg.train.device)
    torch.manual_seed(run_cfg.train.seed)
    np.random.seed(run_cfg.train.seed)

    # graph + nodes
    data = load_pyg_graph(Path(features_run_dir))
    nodes_tab = None
    # load nodes to build node_id map (parquet preferred)
    for name in ("graph_nodes.parquet", "graph_nodes.csv"):
        p = Path(features_run_dir) / name
        if p.exists():
            nodes_tab = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
            break
    if nodes_tab is None:
        raise FileNotFoundError("Could not find graph_nodes.(parquet|csv) next to pyg_graph.pt")

    # trades
    trades = pd.read_parquet(trades_path) if str(trades_path).endswith(".parquet") else pd.read_csv(trades_path)

    # build samples
    samples = build_samples(trades, nodes_tab, run_cfg.sampler)

    if not samples:
        raise RuntimeError("No training samples built. Check grouping and columns in trades.")

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
    ).to(dev)
    data = data.to(dev)  # keep once on device

    # opt/sched
    opt = torch.optim.AdamW(model.parameters(), lr=_safe_float(run_cfg.train.lr, 1e-3),
                            weight_decay=_safe_float(run_cfg.train.weight_decay, 1e-4))
    best = float("inf"); best_ep = -1; patience_left = int(run_cfg.train.patience)

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
            print(f"TensorBoard logging to: {tb_dir}  |  run: tensorboard --logdir {tb_dir.parent}")
        except Exception:
            writer = None

    def beta_schedule(epoch: int, warmup: int = 4, target: float = 0.3, ramp: int = 8) -> float:
        if epoch <= warmup: return 0.0
        e = epoch - warmup
        if e >= ramp: return target
        return target * (e / ramp)

    # train
    global_step = 0
    for ep in range(1, int(run_cfg.train.max_epochs) + 1):
        model.train()
        ep_loss = ep_mae = 0.0; nobs = 0

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

            out = model.forward_from_data(data, tgt, pidx, pbat, pw)
            # mild ramp on q90 weight
            losses = composite_loss(out, r, Lr, alpha=1.0, beta=beta_schedule(ep), gamma=0.1, delta_huber=1.0)

            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            clip_grad_norm_(model.parameters(), float(run_cfg.train.clip_grad))
            opt.step()

            B = int(tgt.numel())
            with torch.no_grad():
                mae = (out["delta_mean"].flatten() - r).abs().mean()
            ep_loss += float(losses["total"].item()) * B
            ep_mae  += float(mae.item()) * B
            nobs    += B

            # TensorBoard step logging
            if writer is not None:
                try:
                    writer.add_scalar("train/loss", float(losses["total"].item()), global_step)
                    writer.add_scalar("train/mae", float(mae.item()), global_step)
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
        val_metrics = eval_epoch(model, data, val_loader, dev)
        val_mae = val_metrics["mae"]

        # log (file)
        with (outdir / "progress.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "epoch": ep,
                "train_mae": train_mae,
                "val": val_metrics
            }) + "\n")

        # TensorBoard epoch logging
        if writer is not None:
            try:
                writer.add_scalar("epoch/train_mae", float(train_mae), ep)
                writer.add_scalar("epoch/val_mae", float(val_mae), ep)
                writer.add_scalar("epoch/val_cov50", float(val_metrics["cov50"]), ep)
                writer.add_scalar("epoch/val_cov90", float(val_metrics["cov90"]), ep)
                writer.add_scalar("epoch/val_width_mean", float(val_metrics["width_mean"]), ep)
                writer.add_scalar("epoch/val_width_std", float(val_metrics["width_std"]), ep)
                # learning rate (first param group)
                try:
                    lr0 = float(opt.param_groups[0].get("lr", float("nan")))
                    writer.add_scalar("epoch/lr", lr0, ep)
                except Exception:
                    pass
            except Exception:
                pass

        print(f"Epoch {ep:02d} | train_mae={train_mae:.4f} | val_mae={val_mae:.4f} | cov50={val_metrics['cov50']:.3f} "
              f"| cov90={val_metrics['cov90']:.3f} | width={val_metrics['width_mean']:.3f}±{val_metrics['width_std']:.3f}")

        # early stop
        if val_mae + 1e-8 < best:
            best = val_mae; best_ep = ep; patience_left = int(run_cfg.train.patience)
            ckpt = {
                "arch": {"type": "liquidity_model_gat",
                         "x_dim": int(data.x.size(1)),
                         "num_relations": num_rel,
                         "model": asdict(run_cfg.model)},
                "state_dict": model.state_dict(),
                "best": {"epoch": best_ep, "val_mae": float(best)},
                "train_config": asdict(run_cfg.train),
            }
            torch.save(ckpt, outdir / "ckpt_liquidity.pt")
            torch.save(ckpt, outdir / "ckpt.pt")
            (outdir / "metrics_val.json").write_text(json.dumps({"best_epoch": best_ep, "best_val_mae_bps": float(best)}, indent=2))
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
    return {"best_epoch": best_ep, "best_val_mae_bps": float(best)}

# Backward-compatible alias if other modules still import the old name
train_liquidity = train_gat
