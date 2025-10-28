from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, Optional

class MLPEncoder(nn.Module):
    """
    Minimal MLP baseline over per-trade baseline features.
    Expects `baseline_feats` tensor of shape [B, baseline_dim].
    Produces a mean prediction and a positive width via Softplus so that
    callers can form q50 and q90 consistently with the GAT-based paths.
    """
    def __init__(self, baseline_dim: int, d_model: int = 128, dropout: float = 0.1):
        super().__init__()
        self.baseline_dim = int(max(1, baseline_dim))
        hid = max(16, d_model // 2)
        self.body = nn.Sequential(
            nn.LayerNorm(self.baseline_dim),
            nn.Linear(self.baseline_dim, hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid, d_model),
            nn.GELU(),
        )
        self.head_mean = nn.Linear(d_model, 1)
        self.head_width = nn.Sequential(nn.Linear(d_model, 1), nn.Softplus())

    def forward(self, baseline_feats: Optional[torch.Tensor], batch_size: Optional[int] = None, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
        """
        If `baseline_feats` is None or empty, create a zero tensor of shape [B, baseline_dim],
        where B is inferred from `batch_size` or 0.
        """
        if baseline_feats is None or (torch.is_tensor(baseline_feats) and baseline_feats.numel() == 0):
            B = int(batch_size or 0)
            x = torch.zeros((B, self.baseline_dim), device=device)
        else:
            x = baseline_feats
            # If provided features have fewer columns (legacy), pad to expected dim
            if x.dim() == 1:
                x = x.unsqueeze(0)
            if x.shape[1] < self.baseline_dim:
                pad = torch.zeros((x.shape[0], self.baseline_dim - x.shape[1]), device=x.device, dtype=x.dtype)
                x = torch.cat([x, pad], dim=1)
        z = self.body(x)
        mean = self.head_mean(z)
        width = self.head_width(z)
        q50 = mean
        q90 = mean + width
        return {"delta_mean": mean, "q50": q50, "q90": q90}
