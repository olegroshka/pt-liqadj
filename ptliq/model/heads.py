from __future__ import annotations
import torch
import torch.nn as nn


class MeanHead(nn.Module):
    def __init__(self, d_in: int, hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QuantileHead(nn.Module):
    """
    Predicts multiple quantiles at once for taus in (0,1).
    """
    def __init__(self, d_in: int, hidden: int, taus: list[float], dropout: float = 0.0):
        super().__init__()
        assert len(taus) >= 1
        self.taus = taus
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, len(taus)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class QuantileHeadPrev(nn.Module):
    def __init__(self, d_in: int, qs=(0.5, 0.9), hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.qs = tuple(qs)
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, len(self.qs))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # [B, len(qs)]


class QuantileHeads(nn.Module):
    """Shared trunk → mean / q50 / q90 heads. Uses softplus gap to keep q90 ≥ q50."""
    def __init__(self, d_model: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(d_model, hidden), nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.mean_head = nn.Linear(hidden, 1)
        self.q50_head  = nn.Linear(hidden, 1)
        self.qgap_head = nn.Linear(hidden, 1)

    def forward(self, z):
        import torch.nn.functional as F
        h = self.trunk(z)
        mean = self.mean_head(h)
        q50  = self.q50_head(h)
        q90  = q50 + F.softplus(self.qgap_head(h))
        return {'delta_mean': mean, 'q50': q50, 'q90': q90}
