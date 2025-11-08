
# gru_loop.py
"""
Minimal, honest GRU baseline for residual drift (Δ) using only
trade-level features + market context sequences.

What it does (and what it does NOT do):
- Uses *only* the parquet files located in --feature_dir:
    train.parquet, val.parquet, test.parquet, market_features.parquet
- Supervised target: y_bps (bps). y_bps is NEVER used as a feature.
- Trade features default:
    ["is_portfolio", "side_sign", "f_size_log", "f_coupon", "f_amount_log",
     "f_sector_code", "f_rating_code", "f_curve_code", "f_days_to_mty"]
  (side_sign is derived from f_side_buy as SELL:+1, BUY:-1)
- Market features: all columns in market_features.parquet except 'asof_date'.
  A fixed window (cfg.model.window) of prior days up to the trade_date is fed
  to a GRU; final hidden state is concatenated with the standardized trade
  feature vector and passed to a small MLP to predict y_bps.
- Computes scalers on TRAIN only, and persists them to outdir.
- Early-stops on validation MAE.
- Saves: ckpt.pt, metrics_val.json, predictions_{val,test}.parquet, and
  small sidecar JSONs with configuration and scalers.

No dependency on MV‑DGT or graph artefacts.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import json
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from tqdm.auto import tqdm  # optional
except Exception:  # pragma: no cover
    tqdm = None


# -----------------------------
# Config
# -----------------------------

@dataclass
class GRUModelConfig:
    hidden: int = 64
    layers: int = 1
    dropout: float = 0.1
    window: int = 5
    # trade columns can be overridden; if None, default list below is used
    trade_cols: Optional[List[str]] = None
    # optional dims for legacy serving/tests
    mkt_dim: Optional[int] = None
    trade_dim: Optional[int] = None

# Backward-compatibility aliases expected by service/tests


@dataclass
class GRUTrainConfig:
    feature_dir: str = "data/features/gru_baseline"
    outdir: str = "models/gru/exp_default"
    device: str = "auto"        # "cpu" | "cuda" | "auto"
    epochs: int = 30
    lr: float = 1e-3
    batch_size: int = 2048
    patience: int = 2           # early stopping on val MAE
    early_stopping: bool = False
    seed: int = 42
    enable_tqdm: bool = True
    model: GRUModelConfig = field(default_factory=GRUModelConfig)


# -----------------------------
# Utils
# -----------------------------

def _resolve_device(dev: str) -> torch.device:
    d = (dev or "cpu").lower()
    if d == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device(d)

def _set_seed(seed: int) -> None:
    seed = int(seed)
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _zscore_np(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    sd_safe = np.where(sd <= 1e-8, 1.0, sd)
    z = (x - mu) / sd_safe
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return z.astype(np.float32)

def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize().dt.date

# -----------------------------
# Model
# -----------------------------

class GRURegressor(nn.Module):
    def __init__(self, mkt_dim: int, trade_dim: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(input_size=mkt_dim, hidden_size=hidden, num_layers=layers, batch_first=True)
        head_in = hidden + trade_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
            nn.Linear(hidden, 1),
        )

    def forward(self, mkt_seq: torch.Tensor, trade_vec: torch.Tensor) -> torch.Tensor:
        # mkt_seq: [B, W, F]; trade_vec: [B, D]
        _, h = self.gru(mkt_seq)
        h_last = h[-1]        # [B, H]
        x = torch.cat([h_last, trade_vec], dim=1)
        y = self.head(x).squeeze(-1)
        return y


# -----------------------------
# Data assembly
# -----------------------------

_DEFAULT_TRADE_COLS = [
    "is_portfolio", "side_sign", "f_size_log", "f_coupon", "f_amount_log",
    "f_sector_code", "f_rating_code", "f_curve_code", "f_days_to_mty"
]

class _Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        mkt_idx_map: Dict[object, int],
        trade_cols: List[str],
        target_col: str = "y_bps",
        date_col: str = "trade_date",
    ):
        self.df = df.reset_index(drop=True).copy()
        self.trade_cols = list(trade_cols)
        self.target_col = target_col
        # map dates → integer row indices in market_features
        tdate = _to_date(self.df[date_col])
        di = []
        for d in tdate:
            if d in mkt_idx_map:
                di.append(int(mkt_idx_map[d]))
            else:
                # fallback: clamp to nearest prior available date
                # (in practice market calendar is constructed to cover train/val/test)
                # use -1 to signal "use earliest row"
                di.append(-1)
        self.date_idx = np.asarray(di, dtype=np.int64)
        # trade matrix
        X = []
        for c in self.trade_cols:
            x = pd.to_numeric(self.df[c], errors="coerce")
            if c == "is_portfolio":
                x = x.astype(float)
            X.append(x.to_numpy())
        self.trade_mat = np.vstack(X).T.astype(np.float32) if X else np.zeros((len(self.df), 0), np.float32)
        # targets
        y = pd.to_numeric(self.df[target_col], errors="coerce").astype(float).to_numpy()
        self.y = y.astype(np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "date_idx": torch.as_tensor(self.date_idx[i], dtype=torch.long),
            "x_trade": torch.as_tensor(self.trade_mat[i], dtype=torch.float32),
            "y": torch.as_tensor(self.y[i], dtype=torch.float32),
        }


def _build_mkt_index(mkt: pd.DataFrame) -> Tuple[Dict[object, int], np.ndarray, List[str]]:
    m = mkt.copy()
    if "asof_date" not in m.columns:
        raise FileNotFoundError("market_features.parquet missing 'asof_date' column")
    m["asof_date"] = _to_date(m["asof_date"])
    m = m.sort_values("asof_date").reset_index(drop=True)
    feat_cols = [c for c in m.columns if c != "asof_date"]
    mat = m[feat_cols].astype(float).to_numpy(np.float32, copy=False)
    idx_map = {d: i for i, d in enumerate(m["asof_date"].tolist())}
    return idx_map, mat, feat_cols


# -----------------------------
# Training
# -----------------------------

def _collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    out = {}
    for k in batch[0].keys():
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out

def _window_sequences(idx: torch.Tensor, mkt_z: torch.Tensor, window: int) -> torch.Tensor:
    """
    idx: [B] integer positions into mkt_z rows (may contain -1 = earliest)
    mkt_z: [T, F]
    returns: [B, W, F]
    """
    B = idx.size(0)
    T, F = mkt_z.size(0), mkt_z.size(1)
    W = max(1, int(window))
    out = torch.zeros((B, W, F), dtype=torch.float32, device=mkt_z.device)
    # convert -1 → 0 (earliest row)
    idx_safe = idx.clone()
    idx_safe[idx_safe < 0] = 0
    for w in range(W):
        t = idx_safe - (W - 1 - w)
        t = t.clamp(0, T - 1)
        out[:, w, :] = mkt_z.index_select(0, t)
    return out

def _metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Tuple[float, float]:
    mae = torch.mean(torch.abs(y_true - y_pred)).item()
    rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()
    return float(mae), float(rmse)

def train_gru(cfg: GRUTrainConfig) -> Dict[str, object]:
    _set_seed(cfg.seed)
    device = _resolve_device(cfg.device)
    feature_dir = Path(cfg.feature_dir)
    outdir = Path(cfg.outdir)
    _ensure_dir(outdir)

    # -------- load parquet files
    train_pq = feature_dir / "train.parquet"
    val_pq   = feature_dir / "val.parquet"
    test_pq  = feature_dir / "test.parquet"
    mkt_pq   = feature_dir / "market_features.parquet"
    for p in [train_pq, val_pq, test_pq, mkt_pq]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    df_tr = pd.read_parquet(train_pq)
    df_va = pd.read_parquet(val_pq)
    df_te = pd.read_parquet(test_pq)
    mkt   = pd.read_parquet(mkt_pq)

    # -------- derive/clean minimal set of features (NO y_bps as input)
    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        # is_portfolio → float {0.,1.}
        if "is_portfolio" in d.columns:
            d["is_portfolio"] = d["is_portfolio"].astype(float)
        else:
            d["is_portfolio"] = 0.0
        # side_sign from f_side_buy (BUY==1 → -1; SELL==0 → +1)
        if "f_side_buy" in d.columns:
            s = pd.to_numeric(d["f_side_buy"], errors="coerce").fillna(0.0).astype(float)
            d["side_sign"] = 1.0 - 2.0 * s
        elif "side_sign" not in d.columns:
            d["side_sign"] = 0.0
        # ensure trade_date exists
        if "trade_date" not in d.columns:
            if "trade_dt" in d.columns:
                d["trade_date"] = d["trade_dt"]
            elif "ts" in d.columns:
                d["trade_date"] = d["ts"]
            else:
                raise ValueError("No trade_date / trade_dt / ts column found")
        d["trade_date"] = pd.to_datetime(d["trade_date"], errors="coerce")
        # Drop rows with missing target for safety
        if "y_bps" not in d.columns:
            raise ValueError("Expected target column 'y_bps' not found")
        d = d.dropna(subset=["y_bps", "trade_date"]).reset_index(drop=True)
        return d

    df_tr = _prepare(df_tr)
    df_va = _prepare(df_va)
    df_te = _prepare(df_te)

    # -------- market matrix & index
    mkt_idx_map, mkt_mat, mkt_feat_names = _build_mkt_index(mkt)

    # -------- default trade columns
    trade_cols = cfg.model.trade_cols if cfg.model.trade_cols is not None else list(_DEFAULT_TRADE_COLS)
    missing = [c for c in trade_cols if c not in df_tr.columns]
    if missing:
        raise ValueError(f"Requested trade columns missing from train: {missing}")

    # -------- datasets
    ds_tr = _Dataset(df_tr, mkt_idx_map, trade_cols, target_col="y_bps", date_col="trade_date")
    ds_va = _Dataset(df_va, mkt_idx_map, trade_cols, target_col="y_bps", date_col="trade_date")
    ds_te = _Dataset(df_te, mkt_idx_map, trade_cols, target_col="y_bps", date_col="trade_date")

    # -------- scalers (TRAIN ONLY)
    trade_mu = ds_tr.trade_mat.mean(axis=0) if ds_tr.trade_mat.size else np.zeros((0,), np.float32)
    trade_sd = ds_tr.trade_mat.std(axis=0) if ds_tr.trade_mat.size else np.ones((0,), np.float32)
    trade_sd = np.where(trade_sd <= 1e-8, 1.0, trade_sd)

    # Market scaler: restrict to TRAIN dates used
    tr_dates = ds_tr.date_idx.copy()
    tr_dates = np.where(tr_dates < 0, 0, tr_dates)
    uniq = np.unique(tr_dates)
    mkt_mu = mkt_mat[uniq].mean(axis=0) if mkt_mat.size else np.zeros((0,), np.float32)
    mkt_sd = mkt_mat[uniq].std(axis=0) if mkt_mat.size else np.ones((0,), np.float32)
    mkt_sd = np.where(mkt_sd <= 1e-8, 1.0, mkt_sd)

    # Persist scalers + feature names for serving
    (outdir / "feature_names.json").write_text(json.dumps({"trade": trade_cols, "market": mkt_feat_names}, indent=2))
    (outdir / "scaler_trade.json").write_text(json.dumps({"mean": trade_mu.tolist(), "std": trade_sd.tolist()}, indent=2))
    (outdir / "scaler_market.json").write_text(json.dumps({"mean": mkt_mu.tolist(), "std": mkt_sd.tolist()}, indent=2))
    (outdir / "config.json").write_text(json.dumps({"train": asdict(cfg), "model": asdict(cfg.model)}, indent=2))

    # -------- tensors on device
    mkt_mu_t = torch.as_tensor(mkt_mu, dtype=torch.float32, device=device)
    mkt_sd_t = torch.as_tensor(mkt_sd, dtype=torch.float32, device=device)

    trade_mu_t = torch.as_tensor(trade_mu, dtype=torch.float32, device=device)
    trade_sd_t = torch.as_tensor(trade_sd, dtype=torch.float32, device=device)

    mkt_all = torch.as_tensor(mkt_mat, dtype=torch.float32, device=device)
    # z-score market rows
    mkt_z = (mkt_all - mkt_mu_t) / torch.where(mkt_sd_t <= 0, torch.ones_like(mkt_sd_t), mkt_sd_t)
    mkt_z = torch.nan_to_num(mkt_z, nan=0.0, posinf=0.0, neginf=0.0)

    # -------- loaders
    tr_loader = torch.utils.data.DataLoader(ds_tr, batch_size=int(cfg.batch_size), shuffle=True, collate_fn=_collate)
    va_loader = torch.utils.data.DataLoader(ds_va, batch_size=int(cfg.batch_size), shuffle=False, collate_fn=_collate)
    te_loader = torch.utils.data.DataLoader(ds_te, batch_size=int(cfg.batch_size), shuffle=False, collate_fn=_collate)

    # -------- model
    mkt_dim = int(mkt_z.size(1))
    trade_dim = int(len(trade_cols))
    model = GRURegressor(mkt_dim=mkt_dim, trade_dim=trade_dim,
                         hidden=int(cfg.model.hidden), layers=int(cfg.model.layers),
                         dropout=float(cfg.model.dropout)).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.lr))

    def _std_trade(x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        return (x - trade_mu_t) / torch.where(trade_sd_t <= 0, torch.ones_like(trade_sd_t), trade_sd_t)

    best = {"mae": math.inf, "rmse": math.inf, "epoch": -1}
    bad = 0
    window = int(cfg.model.window)

    # -------- train loop
    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        it = tr_loader
        if cfg.enable_tqdm and tqdm is not None:
            it = tqdm(tr_loader, desc=f"epoch {epoch}/{cfg.epochs}")
        losses = []
        for batch in it:
            di = batch["date_idx"].to(device)
            x_trade = _std_trade(batch["x_trade"].to(device))
            y = batch["y"].to(device).float()

            mseq = _window_sequences(di, mkt_z, window)  # [B,W,F]
            y_hat = model(mseq, x_trade)

            loss = F.smooth_l1_loss(y_hat, y, beta=1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().item()))
            if cfg.enable_tqdm and tqdm is not None:
                it.set_postfix({"loss": f"{losses[-1]:.4f}"})

        # -------- validation
        model.eval()
        with torch.no_grad():
            y_all = []; p_all = []
            for batch in va_loader:
                di = batch["date_idx"].to(device)
                x_trade = _std_trade(batch["x_trade"].to(device))
                y = batch["y"].to(device).float()
                mseq = _window_sequences(di, mkt_z, window)
                p = model(mseq, x_trade)
                y_all.append(y); p_all.append(p)
            if y_all:
                y_cat = torch.cat(y_all, dim=0)
                p_cat = torch.cat(p_all, dim=0)
                mae, rmse = _metrics(y_cat, p_cat)
            else:
                mae, rmse = float("inf"), float("inf")

        print(f"[GRU] epoch={epoch} train_loss={np.mean(losses):.4f} val_mae_bps={mae:.4f} val_rmse_bps={rmse:.4f}")

        if mae + 1e-12 < best["mae"]:
            best.update({"mae": mae, "rmse": rmse, "epoch": epoch})
            bad = 0
            ckpt = {
                "state_dict": model.state_dict(),
                "arch": {
                    "mkt_dim": mkt_dim,
                    "trade_dim": trade_dim,
                    "hidden": int(cfg.model.hidden),
                    "layers": int(cfg.model.layers),
                    "dropout": float(cfg.model.dropout),
                    "window": int(cfg.model.window),
                },
                "scalers": {
                    "trade": {"mean": trade_mu.tolist(), "std": trade_sd.tolist()},
                    "market": {"mean": mkt_mu.tolist(), "std": mkt_sd.tolist()},
                },
                "feature_names": {"trade": trade_cols, "market": mkt_feat_names},
                "epoch": int(epoch),
                "val_mae_bps": float(mae),
                "val_rmse_bps": float(rmse),
                "units": "bps",
            }
            torch.save(ckpt, Path(outdir) / "ckpt.pt")
        else:
            # Early stopping only if patience > 0; if patience <= 0, run full epochs
            if cfg.early_stopping and int(cfg.patience) > 0:
                bad += 1
                if bad >= int(cfg.patience):
                    print(f"[GRU] Early stopping at epoch {epoch}")
                    break

    # -------- persist metrics + predictions on val/test
    metrics = {
        "best_epoch": int(best["epoch"]),
        "best_val_mae_bps": float(best["mae"]),
        "best_val_rmse_bps": float(best["rmse"]),
        "units": "bps",
    }
    (outdir / "metrics_val.json").write_text(json.dumps(metrics, indent=2))

    # save predictions for sanity checks
    def _dump_preds(loader, df_src: pd.DataFrame, tag: str) -> None:
        model.eval()
        rows = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                di = batch["date_idx"].to(device)
                x_trade = _std_trade(batch["x_trade"].to(device))
                y = batch["y"].to(device).float()
                mseq = _window_sequences(di, mkt_z, window)
                p = model(mseq, x_trade).cpu().numpy()
                yv = y.cpu().numpy()
                # materialize row indices for this slice
                bs = len(yv)
                start = batch_idx * loader.batch_size
                idx = range(start, start + bs)
                for i_local, (yt, yp) in enumerate(zip(yv, p)):
                    rows.append({"y_true_bps": float(yt), "y_pred_bps": float(yp)})
        out_df = pd.concat([df_src.reset_index(drop=True)[["trade_date","isin"]].copy(), pd.DataFrame(rows)], axis=1)
        out_df.to_parquet(Path(outdir) / f"predictions_{tag}.parquet", index=False)

    _dump_preds(va_loader, df_va, "val")
    _dump_preds(te_loader, df_te, "test")

    print(f"[GRU] Training complete. Best epoch={metrics['best_epoch']}, best_val_mae_bps={metrics['best_val_mae_bps']:.4f}")
    return metrics