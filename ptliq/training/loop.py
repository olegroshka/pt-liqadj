from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional, Tuple

import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Config (backward-compatible)
# -----------------------------
@dataclass
class TrainConfig:
    max_epochs: int = 20
    batch_size: int = 512
    lr: float = 1e-3
    patience: int = 3
    device: str = "cpu"        # "cpu" | "cuda" | "auto"
    hidden: List[int] = None   # e.g. [64,64]
    dropout: float = 0.0
    seed: int = 42

    # NEW: future-proofing (does not change current behavior)
    model_type: str = "mlp"          # "mlp" (default) | "gnn_xfmr" (next step)
    model_params: Dict[str, Any] = None  # extra per-model kwargs

    def __post_init__(self):
        if self.hidden is None:
            self.hidden = [64, 64]
        if self.model_params is None:
            self.model_params = {}


# -----------------------------
# Utilities
# -----------------------------
def _resolve_device(dev: str) -> torch.device:
    d = (dev or "cpu").lower()
    if d == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device(d)


def _set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# Models
# -----------------------------
class MLPRegressor(nn.Module):
    """
    Simple MLP: [in_dim] -> hidden* -> 1
    """
    def __init__(self, in_dim: int, hidden: Iterable[int] = (64, 64), dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            if dropout and dropout > 0:
                layers += [nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # [B]


def _build_model(in_dim: int, cfg: TrainConfig) -> nn.Module:
    """
    Factory. 'mlp' is the default (all existing code/tests).
    We also accept 'gnn_xfmr' here so checkpoints can carry model_type,
    but the *training path* for gnn lives in ptliq.training.gnn_loop.
    """
    mtype = (cfg.model_type or "mlp").lower()
    if mtype == "mlp":
        return MLPRegressor(in_dim, hidden=cfg.hidden, dropout=cfg.dropout)
    elif mtype == "gnn_xfmr":
        # We intentionally do not construct the GNN here because that model
        # needs graph tensors rather than a plain feature matrix.
        raise ValueError(
            "model_type='gnn_xfmr' is trained via ptliq.training.gnn_loop.train_gnn ; "
            "use the MLP path for tabular features."
        )
    else:
        raise ValueError(f"Unsupported model_type={cfg.model_type!r}. Use 'mlp' or 'gnn_xfmr'.")


# -----------------------------
# Train / Eval
# -----------------------------
def _iterate_batches(X: np.ndarray, y: np.ndarray, batch_size: int) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    n = X.shape[0]
    for i in range(0, n, batch_size):
        yield X[i:i + batch_size], y[i:i + batch_size]


def train_loop(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    outdir: Path,
    cfg: TrainConfig,
) -> Dict[str, Any]:
    """
    Trains a regressor and writes:
      - ckpt.pt               (state_dict + arch metadata)
      - feature_names.json
    Returns: {"best_epoch": int, "best_val_mae_bps": float}
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "feature_names.json").write_text(json.dumps(feature_names, indent=2))

    _set_seed(cfg.seed)
    dev = _resolve_device(cfg.device)

    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    y_train = y_train.astype(np.float32).reshape(-1)
    y_val = y_val.astype(np.float32).reshape(-1)

    in_dim = X_train.shape[1]
    model = _build_model(in_dim, cfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    best_mae = math.inf
    best_epoch = -1
    bad = 0

    def _mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (pred - target).abs().mean()

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        tr_losses = []
        for xb, yb in _iterate_batches(X_train, y_train, cfg.batch_size):
            xb_t = torch.from_numpy(xb).to(dev)
            yb_t = torch.from_numpy(yb).to(dev)
            pred = model(xb_t)
            loss = F.smooth_l1_loss(pred, yb_t, beta=1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_losses.append(loss.detach().item())

        # validation
        model.eval()
        with torch.no_grad():
            xv = torch.from_numpy(X_val).to(dev)
            yv = torch.from_numpy(y_val).to(dev)
            pv = model(xv)
            mae = _mae(pv, yv).item()

        if mae + 1e-12 < best_mae:
            best_mae = mae
            best_epoch = epoch
            bad = 0
            # save checkpoint with arch metadata to guarantee future loads
            ckpt = {
                "state_dict": model.state_dict(),
                "arch": {
                    "model_type": cfg.model_type,
                    "in_dim": in_dim,
                    "hidden": list(cfg.hidden),
                    "dropout": float(cfg.dropout),
                },
                "epoch": epoch,
                "val_mae": best_mae,
            }
            torch.save(ckpt, outdir / "ckpt.pt")
        else:
            bad += 1

        if bad >= cfg.patience:
            break

    return {"best_epoch": best_epoch, "best_val_mae_bps": float(best_mae)}


def load_model_for_eval(models_dir: Path, in_dim: int, device: str = "cpu") -> Tuple[nn.Module, torch.device]:
    """
    Loads the best checkpoint and returns a ready-to-eval model on the given device.
    Backward compatible:
      - If arch metadata exists, we use it.
      - Else we fall back to (hidden=[64,64], dropout=0.0) with provided in_dim.
    """
    models_dir = Path(models_dir)
    ckpt_path = models_dir / "ckpt.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    arch = ckpt.get("arch", {}) if isinstance(ckpt, dict) else {}
    mtype = arch.get("model_type", "mlp")
    in_dim_ckpt = int(arch.get("in_dim", in_dim))
    hidden = arch.get("hidden", [64, 64])
    dropout = float(arch.get("dropout", 0.0))

    # model factory (only 'mlp' for now)
    cfg = TrainConfig(hidden=hidden, dropout=dropout, device=device, model_type=mtype)
    dev = _resolve_device(cfg.device)
    model = _build_model(in_dim_ckpt, cfg).to(dev)

    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, dev
