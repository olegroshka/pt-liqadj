from __future__ import annotations
import torch
import torch.nn.functional as F


def huber_loss(y_pred: torch.Tensor, y_true: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """
    y_*: [B,1]
    """
    u = y_true - y_pred
    abs_u = torch.abs(u)
    quad = torch.minimum(abs_u, torch.tensor(delta, device=y_true.device))
    lin = abs_u - quad
    return 0.5 * quad**2 + delta * lin


def pinball_loss(yq: torch.Tensor, y_true: torch.Tensor, qs=(0.5, 0.9)) -> torch.Tensor:
    """
    yq: [B, len(qs)]
    y_true: [B,1]
    """
    losses = []
    for i, q in enumerate(qs):
        e = y_true.squeeze(-1) - yq[:, i]
        losses.append(torch.maximum(q * e, (q - 1) * e))
    return torch.stack(losses, dim=1)  # [B, len(qs)]


def monotone_reg(pred: torch.Tensor, ref: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    """
    Soft penalty to encourage d pred / d ref >= 0.
    Approximated via finite differences on batch pairs sorted by ref.
    """
    # skip if no ref provided
    if ref is None:
        return pred.new_tensor(0.0)
    idx = torch.argsort(ref.view(-1))
    p = pred.view(-1)[idx]
    r = ref.view(-1)[idx]
    dp = p[1:] - p[:-1]
    dr = r[1:] - r[:-1]
    viol = torch.clamp(-dp * torch.sign(dr), min=0.0)  # penalize negative slope along increasing ref
    return weight * viol.mean()
