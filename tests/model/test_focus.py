import torch
import pytest

from ptliq.model.portfolio_encoder import PortfolioEncoder


def test_focus_identity_update():
    torch.manual_seed(0)
    # Toy setup: D=3, T=3, batch=1
    d = 3
    enc = PortfolioEncoder(d_model=d, nhead=1, num_layers=1, dropout=0.0, beta_init=0.8, learnable_tau=False)
    enc.eval()

    h = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)                       # (B=1,D=3) = e1
    H = torch.tensor([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=torch.float32)  # (1,3,3)
    valid = torch.tensor([3], dtype=torch.long)

    with torch.no_grad():
        h2 = enc(h, H, valid)
        cos_after = torch.nn.functional.cosine_similarity(h2, torch.tensor([[1.,0.,0.]]), dim=-1)

    # Identity init should keep the target very close to e1 under convex update
    assert torch.all(cos_after > torch.tensor([0.99]))
