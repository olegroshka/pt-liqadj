from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Dict


def pinball_loss(y_hat: torch.Tensor, y: torch.Tensor, tau: float) -> torch.Tensor:
    u = y - y_hat
    return torch.mean(torch.maximum(torch.tensor(tau, dtype=y.dtype, device=y.device)*u,
                                    (torch.tensor(tau, dtype=y.dtype, device=y.device)-1)*u))


def huber_loss(y_hat: torch.Tensor, y: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    u = y - y_hat
    abs_u = torch.abs(u)
    delta_t = torch.tensor(delta, dtype=y.dtype, device=y.device)
    quad  = torch.minimum(abs_u, delta_t)
    return torch.mean(0.5*quad**2 + delta_t*(abs_u - quad))


def monotonicity_regularizer(lref: torch.Tensor, delta_pred: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    """Pairwise hinge: as liquidity proxy L increases, predicted mean residual should not decrease."""
    l = lref.view(-1); d = delta_pred.view(-1)
    D_L = l.unsqueeze(1) - l.unsqueeze(0)   # (B,B): L_j - L_i
    mask = (D_L > 0)                        # j is more illiquid than i
    if not mask.any():
        return torch.zeros((), device=l.device)
    D_D = d.unsqueeze(1) - d.unsqueeze(0)   # (B,B): d_j - d_i
    viol = torch.relu(torch.tensor(margin, device=d.device, dtype=d.dtype) - D_D)[mask]
    return viol.mean()


def noncross_penalty(q50: torch.Tensor, q90: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    return torch.relu(q50 - q90 + eps).mean()


def composite_loss(
    pred: Dict[str, torch.Tensor],
    r_true: torch.Tensor,
    lref_target: torch.Tensor,
    # core weights
    alpha: float = 1.0,       # q50 pinball
    beta: float  = 0.8,       # q90 pinball
    gamma: float = 0.1,       # monotonicity of mean vs L
    delta_huber: float = 1.0,
    lambda_huber: float = 1.0, # weight for Huber term
    # guardrails
    lambda_noncross: float = 0.10,
    noncross_eps: float = 1e-4,
    lambda_wmono: float = 0.10,  # width non-decreasing in L
    wmono_margin: float = 0.0,
) -> Dict[str, torch.Tensor]:
    r = r_true.view(-1, 1)
    L = lref_target.view(-1, 1)

    huber = huber_loss(pred['delta_mean'], r, delta=delta_huber)
    q50   = pinball_loss(pred['q50'], r, tau=0.5)
    q90   = pinball_loss(pred['q90'], r, tau=0.9)
    mono  = monotonicity_regularizer(L, pred['delta_mean'], margin=0.0)

    # non-crossing
    nc = noncross_penalty(pred['q50'], pred['q90'], eps=noncross_eps)

    # width monotonicity in-batch
    W  = (pred['q90'] - pred['q50']).clamp_min(0.0)
    DL = L.t() - L
    DW = W.t() - W
    mask = (DL > 0)
    wmono = torch.relu(torch.tensor(wmono_margin, dtype=W.dtype, device=W.device) - DW)[mask].mean() if mask.any() else torch.zeros((), device=r.device)

    total = (lambda_huber * huber) + alpha*q50 + beta*q90 + gamma*mono + lambda_noncross*nc + lambda_wmono*wmono
    return {
        'total': total, 'huber': huber, 'q50': q50, 'q90': q90,
        'mono': mono, 'noncross': nc, 'wpen': wmono, 'width_mean': W.mean()
    }
