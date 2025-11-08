import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from ptliq.training.gru_loop import GRURegressor


def test_gru_quick_learn_cpu():
    # Synthetic sequence regression: y depends linearly on last hidden (market) and trade features
    torch.manual_seed(0)
    np.random.seed(0)

    B = 256
    W = 3
    F = 5
    trade_dim = 2

    # Generate market sequences and trade features
    mseq = torch.randn(B, W, F)
    trade = torch.randn(B, trade_dim)

    # True weights
    Wh = torch.randn(F) * 0.4
    Wt = torch.tensor([0.8, -0.3])

    # Target: dot of last step with Wh + trade @ Wt + noise
    y = (mseq[:, -1, :] @ Wh) + (trade @ Wt) + torch.randn(B) * 0.05

    model = GRURegressor(mkt_dim=F, trade_dim=trade_dim, hidden=16, layers=1, dropout=0.0)
    opt = optim.Adam(model.parameters(), lr=5e-3)

    # Train
    steps = 400
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        pred = model(mseq, trade)
        loss = nn.functional.l1_loss(pred, y)
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred = model(mseq, trade)
        mae = torch.mean(torch.abs(pred - y)).item()

    assert mae < 0.15, f"GRU quick-learn MAE too high: {mae:.4f}"
