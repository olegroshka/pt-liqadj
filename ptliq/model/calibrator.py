from __future__ import annotations
import torch
import torch.nn as nn


class MonotoneCalibrator(nn.Module):
    """
    Provider-specific monotone calibration via non-negative piecewise-linear spline.
    y = a0 + sum_{k} softplus(w_k) * ReLU(x - t_k)
    """
    def __init__(self, n_knots: int = 8):
        super().__init__()
        self.a0 = nn.Parameter(torch.zeros(1))
        self.t = nn.Parameter(torch.linspace(0, 1, steps=n_knots))  # knots in normalized vendor space
        self.w = nn.Parameter(torch.zeros(n_knots))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x expected roughly normalized to [0,1]
        deltas = torch.relu(x[..., None] - self.t)  # [..., K]
        slope = torch.nn.functional.softplus(self.w)  # non-negative
        return self.a0 + (deltas * slope).sum(-1, keepdim=True)
