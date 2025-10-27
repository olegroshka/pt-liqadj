# ptliq/training/gnn_loop.py
from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import torch
from torch import nn

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
def configure_logging(level: str = "INFO") -> None:
    if LOG.handlers:
        return
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    )

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _pick_device(device: str) -> torch.device:
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    LOG.info("Using device: %s", dev)
    return dev

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    LOG.info("Seed set to %d", seed)

def _index_any(obj: Any, idx: Union[slice, torch.Tensor]) -> Any:
    if obj is None:
        return None
    if torch.is_tensor(obj):
        return obj[idx]
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            out[k] = v[idx] if torch.is_tensor(v) else v
        return out
    return obj

def _nums_first_tensor(nums: Any) -> torch.Tensor:
    if torch.is_tensor(nums):
        return nums
    if isinstance(nums, dict):
        for v in nums.values():
            if torch.is_tensor(v):
                return v
    raise RuntimeError("Cannot find a tensor inside 'nums'.")

def _infer_num_dim(nums: Any) -> int:
    """Return feature dim for `nums`, robust to [N] vs [N,D] tensors and dicts."""
    if torch.is_tensor(nums):
        return 1 if nums.dim() == 1 else int(nums.shape[1])
    if isinstance(nums, dict):
        D = 0
        for v in nums.values():
            if torch.is_tensor(v):
                D += (1 if v.dim() == 1 else int(v.shape[1]))
        if D > 0:
            return D
    raise RuntimeError("Cannot infer num dim: 'nums' has no tensor.")

def _nums_to_tensor(nums: Any) -> torch.Tensor:
    """
    Coerce numeric features to a single [B, D] tensor.
    - Tensor [B] -> [B,1]
    - Tensor [B,D] -> [B,D]
    - dict[str -> Tensor[B] or Tensor[B,1]] -> cat(sorted keys) -> [B,D]
    """
    if torch.is_tensor(nums):
        return nums.unsqueeze(-1) if nums.dim() == 1 else nums
    if isinstance(nums, dict):
        if not nums:
            raise RuntimeError("nums dict is empty; cannot build numeric tensor.")
        keys = sorted(nums.keys())
        cols = []
        B = None
        for k in keys:
            t = nums[k]
            if not torch.is_tensor(t):
                raise RuntimeError(f"nums[{k}] is not a tensor")
            if B is None:
                B = int(t.shape[0])
            if t.dim() == 1:
                t = t.unsqueeze(-1)  # [B] -> [B,1]
            cols.append(t)
        return torch.cat(cols, dim=-1) if cols else torch.zeros((B or 0, 0))
    raise RuntimeError(f"Unsupported 'nums' type: {type(nums)}")

# Import GraphInputs safely
try:
    from ptliq.model.utils import GraphInputs  # noqa: F401
except Exception:
    GraphInputs = Any  # type: ignore

# ---------------------------------------------------------------------
# Config / Interfaces
# ---------------------------------------------------------------------
@dataclass
class GNNTrainConfig:
    device: str = "auto"
    max_epochs: int = 20
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 3
    seed: int = 42

    node_id_dim: int = 32
    nhead: int = 4
    n_layers: int = 1
    d_model: int = 128
    gnn_num_hidden: int = 64
    gnn_out_dim: int = 128
    gnn_dropout: float = 0.0
    head_hidden: int = 128
    head_dropout: float = 0.0
    use_calibrator: bool = False
    use_baseline: bool = False

    log_level: str = "INFO"

class PortfolioResidualModelLike(nn.Module):
    """
    forward(
      node_ids, cats, nums,
      issuer_groups, sector_groups,
      node_to_issuer, node_to_sector,
      port_nodes=None, port_len=None,
      size_side_urg=None, baseline_feats=None
    ) -> {"mean": Tensor[N,1] or [N]}
    """
    pass

class TinyLinearModel(PortfolioResidualModelLike):
    def __init__(self, in_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, 1)

    def forward(
        self,
        node_ids, cats, nums,
        issuer_groups, sector_groups,
        node_to_issuer, node_to_sector,
        port_nodes=None, port_len=None,
        size_side_urg=None, baseline_feats=None,
    ):
        x = _nums_first_tensor(nums)
        return {"mean": self.lin(x)}

# ---------------------------------------------------------------------
# Batch packing
# ---------------------------------------------------------------------
def _pack_batch(gi: GraphInputs, sl: slice) -> Dict[str, Any]:
    batch = {
        "node_ids": gi.node_ids[sl],
        "cats": {k: v[sl] for k, v in gi.cats.items()},
        "nums": _index_any(gi.nums, sl),
        "issuer_groups": gi.issuer_groups,
        "sector_groups": gi.sector_groups,
        "node_to_issuer": gi.node_to_issuer,
        "node_to_sector": gi.node_to_sector,
        "port_nodes": gi.port_nodes[sl] if getattr(gi, "port_nodes", None) is not None and gi.port_nodes.numel() > 0 else getattr(gi, "port_nodes", None),
        "port_len": gi.port_len[sl] if getattr(gi, "port_len", None) is not None and gi.port_len.numel() > 0 else getattr(gi, "port_len", None),
        "y": gi.y[sl],
    }
    if hasattr(gi, "size_side_urg") and gi.size_side_urg is not None:
        ssu = gi.size_side_urg
        batch["size_side_urg"] = ssu[sl] if torch.is_tensor(ssu) and ssu.numel() else ssu
    else:
        batch["size_side_urg"] = None
    if hasattr(gi, "baseline_feats") and gi.baseline_feats is not None:
        bf = gi.baseline_feats
        batch["baseline_feats"] = bf[sl] if torch.is_tensor(bf) and bf.numel() else bf
    else:
        batch["baseline_feats"] = None
    return batch

def _pack_indexed(gi: GraphInputs, idx: torch.Tensor) -> Dict[str, Any]:
    """Pack a batch by global indices (avoids earlier permute-slice bug)."""
    batch = {
        "node_ids": gi.node_ids[idx],
        "cats": {k: v[idx] for k, v in gi.cats.items()},
        "nums": _index_any(gi.nums, idx),
        "issuer_groups": gi.issuer_groups,
        "sector_groups": gi.sector_groups,
        "node_to_issuer": gi.node_to_issuer,
        "node_to_sector": gi.node_to_sector,
        "port_nodes": gi.port_nodes[idx] if getattr(gi, "port_nodes", None) is not None and gi.port_nodes.numel() > 0 else getattr(gi, "port_nodes", None),
        "port_len": gi.port_len[idx] if getattr(gi, "port_len", None) is not None and gi.port_len.numel() > 0 else getattr(gi, "port_len", None),
        "y": gi.y[idx],
    }
    if hasattr(gi, "size_side_urg") and torch.is_tensor(getattr(gi, "size_side_urg", None)) and gi.size_side_urg.numel() > 0:
        batch["size_side_urg"] = gi.size_side_urg[idx]
    else:
        batch["size_side_urg"] = None
    if hasattr(gi, "baseline_feats") and torch.is_tensor(getattr(gi, "baseline_feats", None)) and gi.baseline_feats.numel() > 0:
        batch["baseline_feats"] = gi.baseline_feats[idx]
    else:
        batch["baseline_feats"] = None
    return batch

def _to_dev(batch: Dict[str, Any], dev: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(dev, non_blocking=True)
        elif isinstance(v, dict):
            out[k] = {kk: vv.to(dev, non_blocking=True) if torch.is_tensor(vv) else vv for kk, vv in v.items()}
        else:
            out[k] = v
    return out

# ---------------------------------------------------------------------
# Loss / metric
# ---------------------------------------------------------------------
def _mae(y: torch.Tensor, yhat: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(y - yhat))

def _rmse(y: torch.Tensor, yhat: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((y - yhat) ** 2))

# ---------------------------------------------------------------------
# Checkpoint metadata: serialize to pure primitives (safe for weights_only loads)
# ---------------------------------------------------------------------
def _to_primitives_model_config(mcfg: Any) -> Dict[str, Any]:
    d = dict(mcfg.__dict__) if hasattr(mcfg, "__dict__") else {}
    cat_specs = []
    for spec in d.get("cat_specs", []) or []:
        if isinstance(spec, dict):
            cat_specs.append({k: spec[k] for k in ("name", "n", "dim") if k in spec})
        else:
            name = getattr(spec, "name", None)
            n = getattr(spec, "n", getattr(spec, "num_categories", None))
            dim = getattr(spec, "dim", getattr(spec, "emb_dim", None))
            cat_specs.append({"name": name, "n": int(n) if n is not None else None, "dim": int(dim) if dim is not None else None})
    d["cat_specs"] = cat_specs
    for k, v in list(d.items()):
        if isinstance(v, (np.generic,)):
            d[k] = v.item()
        elif isinstance(v, (Path,)):
            d[k] = str(v)
        elif not isinstance(v, (str, int, float, bool, list, dict, type(None))):
            d[k] = str(v)
    return d

# ---------------------------------------------------------------------
# Core training loop
# ---------------------------------------------------------------------
def train_gnn(
    train_gi: GraphInputs,
    val_gi: GraphInputs,
    outdir: Path,
    cfg: GNNTrainConfig,
    model_factory: Optional[Callable[[GraphInputs, GNNTrainConfig], PortfolioResidualModelLike]] = None,
) -> Dict[str, Any]:
    """
    Train a model with the minimal PortfolioResidualModelLike contract.

    Saves:
      - {outdir}/ckpt_gnn.pt
      - {outdir}/ckpt.pt
      - {outdir}/metrics_val.json
      - {outdir}/progress.jsonl
    """
    configure_logging(cfg.log_level)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "progress.jsonl").touch()

    _set_seed(cfg.seed)
    dev = _pick_device(cfg.device)

    # Build default model unless a factory is provided.
    if model_factory is None:
        from ptliq.model import PortfolioResidualModel, ModelConfig, NodeFieldSpec  # lazy import
        n_num = _infer_num_dim(train_gi.nums)
        LOG.debug("Detected n_num=%d (type=%s)", n_num, type(train_gi.nums).__name__)
        cat_specs = []
        if "sector_code" in train_gi.cats:
            vmax = int(train_gi.cats["sector_code"].max().item()) + 1
            cat_specs.append(NodeFieldSpec("sector_code", vmax, 16))
        if "rating_code" in train_gi.cats:
            vmax = int(train_gi.cats["rating_code"].max().item()) + 1
            cat_specs.append(NodeFieldSpec("rating_code", vmax, 12))

        mcfg = ModelConfig(
            n_nodes=train_gi.n_nodes,
            node_id_dim=int(cfg.node_id_dim),
            cat_specs=cat_specs,
            n_num=n_num,
            gnn_num_hidden=int(cfg.gnn_num_hidden),
            gnn_out_dim=int(cfg.gnn_out_dim),
            d_model=int(cfg.d_model),
            nhead=int(cfg.nhead),
            n_layers=int(cfg.n_layers),
            device=cfg.device,
            use_baseline=bool(cfg.use_baseline),
        )
        model: PortfolioResidualModelLike = PortfolioResidualModel(mcfg).to(dev)
        arch_meta = {"type": "portfolio_residual_model", "model_config": _to_primitives_model_config(mcfg)}
    else:
        model = model_factory(train_gi, cfg).to(dev)
        arch_meta = {"type": "factory_model", "hint": model.__class__.__name__}

    # quick baseline: mean predictor on train, evaluated on val
    with torch.no_grad():
        y_tr_mean = float(train_gi.y.mean().item())
        y_va = val_gi.y
        baseline_mae = float(torch.mean(torch.abs(y_va - y_tr_mean)).item())
    LOG.info("Baseline (mean predictor on train, MAE on val): %.4f", baseline_mae)

    # Log model
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOG.info("Model: %s (trainable params: %s)", model.__class__.__name__, f"{n_params:_}")
    LOG.debug("Model structure:\n%s", model)

    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    best_val = float("inf")
    best_epoch = -1
    history: Dict[str, list[float]] = {"val_mae_bps": [], "train_mae_bps": [], "val_rmse_bps": [], "train_rmse_bps": []}
    patience_left = int(cfg.patience)

    n_train = int(train_gi.node_ids.shape[0])
    bs = int(cfg.batch_size)
    LOG.info(
        "Train size: %d | Val size: %d | Batch size: %d | Max epochs: %d",
        n_train, int(val_gi.node_ids.shape[0]), bs, int(cfg.max_epochs)
    )

    def _normalize_port_inputs(b: Dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Ensure shapes: port_nodes [B,T], port_len [B]. Handle empty tensors."""
        pn = b.get("port_nodes", None)
        pl = b.get("port_len", None)
        B = int(b["node_ids"].shape[0])

        if pn is None or (torch.is_tensor(pn) and pn.numel() == 0):
            pn = torch.zeros((B, 0), dtype=torch.long, device=b["node_ids"].device)
        elif pn.dim() == 1:
            pn = pn.unsqueeze(1)

        if pl is None or (torch.is_tensor(pl) and pl.numel() == 0):
            pl = torch.zeros((B,), dtype=torch.long, device=b["node_ids"].device)

        return pn, pl

    def _eval_split(gi: GraphInputs) -> Dict[str, float]:
        model.eval()
        with torch.no_grad():
            N = int(gi.node_ids.shape[0])
            tot_mae = 0.0
            tot_mse = 0.0
            cnt = 0
            for start in range(0, N, bs):
                end = min(start + bs, N)
                b = _pack_batch(gi, slice(start, end))
                b = _to_dev(b, dev)

                size_side_urg = b.get("size_side_urg")
                baseline_feats = b.get("baseline_feats")
                if size_side_urg is None:
                    size_side_urg = torch.zeros((b["node_ids"].shape[0], 3), device=dev)
                if baseline_feats is None:
                    baseline_feats = torch.zeros((b["node_ids"].shape[0], 0), device=dev)

                pn, pl = _normalize_port_inputs(b)
                nums_tensor = _nums_to_tensor(b["nums"])

                out = model(
                    b["node_ids"], b["cats"], nums_tensor,
                    b["issuer_groups"], b["sector_groups"],
                    b["node_to_issuer"], b["node_to_sector"],
                    pn, pl,
                    size_side_urg, baseline_feats,
                )
                yhat = out["mean"]
                if yhat.dim() > 1 and yhat.shape[-1] == 1:
                    yhat = yhat.squeeze(-1)
                y_true = b["y"]
                if y_true.dim() > 1 and y_true.shape[-1] == 1:
                    y_true = y_true.squeeze(-1)
                err = (y_true - yhat)
                mae = torch.mean(torch.abs(err)).item()
                mse = torch.mean(err * err).item()
                tot_mae += float(mae) * (end - start)
                tot_mse += float(mse) * (end - start)
                cnt += (end - start)
            mean_mae = tot_mae / max(cnt, 1)
            mean_rmse = float(np.sqrt(tot_mse / max(cnt, 1)))
            return {"mae": mean_mae, "rmse": mean_rmse}

    for epoch in range(int(cfg.max_epochs)):
        model.train()
        perm = torch.randperm(n_train)
        running_loss = 0.0
        running_mae = 0.0
        n_seen = 0

        for start in range(0, n_train, bs):
            end = min(start + bs, n_train)
            idx = perm[start:end]                    # GLOBAL indices
            b = _pack_indexed(train_gi, idx)        # pack by indices
            b = _to_dev(b, dev)

            size_side_urg = b.get("size_side_urg")
            baseline_feats = b.get("baseline_feats")
            if size_side_urg is None:
                size_side_urg = torch.zeros((b["node_ids"].shape[0], 3), device=dev)
            if baseline_feats is None:
                baseline_feats = torch.zeros((b["node_ids"].shape[0], 0), device=dev)

            pn, pl = _normalize_port_inputs(b)
            nums_tensor = _nums_to_tensor(b["nums"])

            out = model(
                b["node_ids"], b["cats"], nums_tensor,
                b["issuer_groups"], b["sector_groups"],
                b["node_to_issuer"], b["node_to_sector"],
                pn, pl,
                size_side_urg, baseline_feats,
            )

            yhat = out["mean"]
            if yhat.dim() > 1 and yhat.shape[-1] == 1:
                yhat = yhat.squeeze(-1)
            y_true = b["y"]
            if y_true.dim() > 1 and y_true.shape[-1] == 1:
                y_true = y_true.squeeze(-1)

            loss = nn.functional.mse_loss(yhat, y_true)
            mae_batch = _mae(y_true, yhat)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running_loss += float(loss.item()) * (end - start)
            running_mae += float(mae_batch.item()) * (end - start)
            n_seen += (end - start)

        train_mae = running_mae / max(n_seen, 1)
        avg_loss = running_loss / max(n_seen, 1)
        train_rmse = float(np.sqrt(avg_loss))
        val_metrics = _eval_split(val_gi)
        val_mae = float(val_metrics["mae"])
        val_rmse = float(val_metrics["rmse"])
        history["train_mae_bps"].append(train_mae)
        history["val_mae_bps"].append(val_mae)
        history["train_rmse_bps"].append(train_rmse)
        history["val_rmse_bps"].append(val_rmse)

        LOG.info(
            "Epoch %3d/%d | train_mae=%.4f | train_rmse=%.4f | val_mae=%.4f | val_rmse=%.4f | loss=%.4f",
            epoch + 1, int(cfg.max_epochs), train_mae, train_rmse, val_mae, val_rmse, avg_loss
        )
        with (outdir / "progress.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "epoch": epoch + 1,
                "baseline_mae": baseline_mae,
                "train_mae": train_mae,
                "train_rmse": train_rmse,
                "val_mae": val_mae,
                "val_rmse": val_rmse,
                "avg_loss": avg_loss
            }) + "\n")

        improved = val_mae + 1e-8 < best_val
        if improved:
            best_val = val_mae
            best_epoch = epoch
            patience_left = int(cfg.patience)
            ckpt = {
                "arch": arch_meta,                                  # pure primitives
                "state_dict": model.state_dict(),
                "best": {"best_epoch": best_epoch, "best_val_mae_bps": float(best_val), "best_val_rmse_bps": float(val_rmse), "history": history},
                "train_config": asdict(cfg),
            }
            torch.save(ckpt, outdir / "ckpt_gnn.pt")
            torch.save(ckpt, outdir / "ckpt.pt")
            (outdir / "metrics_val.json").write_text(json.dumps(ckpt["best"], indent=2))
            LOG.info("New best at epoch %d (val_mae=%.4f, val_rmse=%.4f). Checkpoint saved.", epoch + 1, best_val, val_rmse)
        else:
            patience_left -= 1
            LOG.info("No improvement. Patience left: %d", patience_left)
            if patience_left <= 0:
                LOG.info("Early stopping at epoch %d.", epoch + 1)
                break

    if best_epoch < 0:
        ckpt = {
            "arch": arch_meta,
            "state_dict": model.state_dict(),
            "best": {"best_epoch": -1, "best_val_mae_bps": float(best_val), "history": history},
            "train_config": asdict(cfg),
        }
        torch.save(ckpt, outdir / "ckpt_gnn.pt")
        torch.save(ckpt, outdir / "ckpt.pt")
        (outdir / "metrics_val.json").write_text(json.dumps(ckpt["best"], indent=2))

    return {"best_epoch": best_epoch, "best_val_mae_bps": float(best_val), "history": history}
