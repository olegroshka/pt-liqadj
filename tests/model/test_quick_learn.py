import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from ptliq.model.portfolio_encoder import PortfolioEncoder


def _make_synth_batch(B=64, T=8, D=16, noise_bps=1.0, beta_init=0.8):
    torch.manual_seed(0)
    np.random.seed(0)
    enc = PortfolioEncoder(d_model=D, nhead=1, num_layers=1, dropout=0.0, beta_init=beta_init, learnable_tau=False)
    enc.eval()
    # random targets and portfolios with padding length T
    h = torch.randn(B, D)
    H = torch.randn(B, T, D)
    # valid lengths: all valid here for simplicity
    valid = torch.full((B,), T, dtype=torch.long)

    with torch.no_grad():
        # Compute encoder outputs to define a linear readout target
        out = enc(h, H, valid)             # (B, D)
        # Plant a linear readout with small weights (bps scale)
        w = torch.randn(D) * 0.05
        y = (out @ w).detach() * 10.0      # scale to ~bps
        # Add tiny Gaussian noise (bps)
        y = y + torch.randn_like(y) * (noise_bps / 10.0)

    # Freeze encoder for the training step
    for p in enc.parameters():
        p.requires_grad_(False)

    return enc, h, H, valid, y


def test_quick_learn_cpu():
    # Small synthetic problem: head should fit in <= 300 steps to MAE < 5 bps
    B, T, D = 96, 8, 16
    enc, h, H, valid, y = _make_synth_batch(B=B, T=T, D=D, noise_bps=1.0)

    head = nn.Sequential(
        nn.Linear(D, 32), nn.ReLU(),
        nn.Linear(32, 1)
    )
    opt = optim.Adam(head.parameters(), lr=5e-3)

    y = y.view(-1, 1)
    steps = 300
    for t in range(steps):
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            z = enc(h, H, valid)       # (B,D)
        pred = head(z)
        loss = nn.functional.l1_loss(pred, y)
        loss.backward()
        opt.step()

    with torch.no_grad():
        z = enc(h, H, valid)
        pred = head(z)
        mae = torch.mean(torch.abs(pred - y)).item() * 1.0

    assert mae < 5.0, f"Quick-learn MAE too high: {mae:.3f} bps"