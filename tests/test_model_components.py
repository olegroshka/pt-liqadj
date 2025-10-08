import torch
from ptliq.model.portfolio_encoder import PortfolioEncoder


def test_portfolio_encoder_masks_padding():
    torch.manual_seed(0)
    B, T, D = 2, 5, 16
    enc = PortfolioEncoder(d_model=D, nhead=4, num_layers=1, dropout=0.0)

    # Build a portfolio where only the first 2 items are valid; the rest are zeros
    H_p = torch.randn(B, T, D)
    H_p[:, 2:] = 0.0  # padded positions should be ignored
    valid_len = torch.tensor([2, 2])

    h_t = torch.randn(B, D)
    out = enc(h_t, H_p, valid_len)

    # Now perturb ONLY the padded tail; fused output should not change materially
    H_p_perturbed = H_p.clone()
    H_p_perturbed[:, 2:] = torch.randn_like(H_p_perturbed[:, 2:]) * 10.0  # large noise on pads
    out2 = enc(h_t, H_p_perturbed, valid_len)

    diff = (out - out2).abs().max().item()
    assert diff < 1e-5, f"padding mask ineffective; diff={diff}"


def test_portfolio_encoder_focus_signal():
    torch.manual_seed(1)
    B, T, D = 2, 6, 32
    enc = PortfolioEncoder(d_model=D, nhead=4, num_layers=1, dropout=0.0)

    # Create portfolios where item 3 encodes the signal; others are noise.
    H_p = torch.randn(B, T, D) * 0.1
    signal = torch.randn(B, D)
    H_p[:, 3] = signal  # the key item
    valid_len = torch.full((B,), T, dtype=torch.long)

    # Query roughly aligned with signal
    h_t = signal + 0.05 * torch.randn(B, D)

    out = enc(h_t, H_p, valid_len)

    # Out should be closer to signal than the raw h_t (attention should pull towards the matching key)
    d0 = (h_t - signal).pow(2).sum(dim=1).mean().item()
    d1 = (out - signal).pow(2).sum(dim=1).mean().item()
    assert d1 < d0, f"cross-attention did not move query towards the signal (d1={d1:.4f} >= d0={d0:.4f})"
