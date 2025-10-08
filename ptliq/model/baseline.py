from __future__ import annotations
import torch
from torch import nn

class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int] = [64, 64], dropout: float = 0.0):
        super().__init__()
        layers = []
        dim = in_dim
        for h in hidden:
            layers += [nn.Linear(dim, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = h
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # [B]


class SizeUrgencyBaseline(nn.Module):
    """
    Small MLP on (log size, side, urgency).
    """
    def __init__(self, d_in: int = 3, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # [B,1]