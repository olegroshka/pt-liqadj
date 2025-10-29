from __future__ import annotations
from typing import Dict, Tuple, Optional

import torch

try:  # pragma: no cover - tensorboard is optional
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:  # pragma: no cover - tensorboard is optional
    SummaryWriter = None  # type: ignore


def extract_heads_mean_std(alpha: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize attention weights tensor `alpha` to shape [E, H] and compute per-head
    mean and std over edges.
    Returns (mean, std) on the SAME device as `alpha` to avoid GPU↔CPU sync during training.
    """
    # alpha may be [E], [E, H], or [H, E]; normalize to [E, H]
    if alpha.dim() == 1:
        alpha_ = alpha.view(-1, 1)
    elif alpha.dim() == 2:
        if alpha.size(0) <= alpha.size(1):
            alpha_ = alpha  # [E, H]
        else:
            alpha_ = alpha.t()  # [E, H]
    else:
        alpha_ = alpha.reshape(alpha.size(0), -1)
    # Compute on-device to avoid implicit synchronizations
    with torch.no_grad():
        alpha_f = alpha_.float()
        mean = alpha_f.mean(dim=0)
        std = alpha_f.std(dim=0, unbiased=False)
    return mean, std


def attn_store_update(store: Dict, layer_key: str, view: str, mean: torch.Tensor, std: torch.Tensor) -> None:
    """Update nested attention stats dict with mean/std tensors for a given layer/view."""
    store.setdefault(layer_key, {})[view] = {"mean": mean, "std": std}


def enable_model_attn_capture(model) -> None:
    """
    Enable attention capture on the model by setting flags and initializing the store.
    This assumes the model reads `self._capture_attn` and writes to `self._attn_stats`.
    """
    try:
        setattr(model, "_capture_attn", True)
        setattr(model, "_attn_stats", {})
    except Exception:
        # leave silently if model doesn't support it
        pass


def disable_model_attn_capture(model) -> None:
    """Disable attention capture and clear the temporary store."""
    try:
        setattr(model, "_capture_attn", False)
        setattr(model, "_attn_stats", {})
    except Exception:
        pass


def log_attn_tb(writer: Optional["SummaryWriter"], stats: Dict, global_step: int, prefix: str = "attn/") -> None:
    """
    Log attention head statistics into TensorBoard. Expects `stats` as produced by
    the model (nested dict: layer_key -> view -> {mean, std}).
    No-op when writer is None or SummaryWriter unavailable.

    Performance notes:
    - Move tensors to CPU once per vector using `.detach().cpu().tolist()` to avoid
      many small GPU→CPU synchronizations from repeated `.item()` calls.
    - Keep training-time reductions on GPU; only transfer small per-head vectors.
    """
    if writer is None or SummaryWriter is None:
        return
    try:
        for layer_key, views in (stats or {}).items():
            for view_name, d in (views or {}).items():
                mean_t = d.get("mean")
                std_t = d.get("std")
                if mean_t is None or std_t is None:
                    continue
                # Single sync per tensor
                mean_list = mean_t.detach().float().cpu().tolist()
                std_list = std_t.detach().float().cpu().tolist()
                for h, (m, s) in enumerate(zip(mean_list, std_list)):
                    writer.add_scalar(f"{prefix}{layer_key}/{view_name}/head{h}_mean", float(m), global_step)
                    writer.add_scalar(f"{prefix}{layer_key}/{view_name}/head{h}_std", float(s), global_step)
    except Exception:
        # swallow TB/logging errors to avoid training interruption
        pass
