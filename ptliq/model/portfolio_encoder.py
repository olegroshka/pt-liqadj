from __future__ import annotations
import torch
import torch.nn as nn


class PortfolioEncoder(nn.Module):
    """
    Single-step cross-attention from a trade query h_t (B, D)
    over a portfolio sequence H_p (B, T, D) with key-padding mask.

    Returns a context vector (B, D).
    """

    def __init__(self, d_model: int, nhead: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        # We keep one effective cross-attn layer; stacking with residuals & norms is usually enough here.
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True, dropout=dropout)
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.ln_out = nn.LayerNorm(d_model)
        # Learnable attention temperature (multiplies logits after the internal 1/sqrt(d))
        self.log_tau = nn.Parameter(torch.zeros(()))  # tau = exp(log_tau) \in (0, inf)

    @staticmethod
    def _key_padding_mask(valid_len: torch.Tensor, T: int) -> torch.Tensor:
        """
        valid_len: (B,) number of valid positions per row.
        Returns mask (B, T) with True where positions are PAD (should be ignored by attention).
        """
        device = valid_len.device
        idx = torch.arange(T, device=device).unsqueeze(0).expand(valid_len.shape[0], T)
        valid = idx < valid_len.unsqueeze(1)  # True for valid tokens
        pad_mask = ~valid                      # True for PAD
        return pad_mask

    def forward(self, h_t: torch.Tensor, H_p: torch.Tensor, valid_len: torch.Tensor) -> torch.Tensor:
        """
        h_t: (B, D)
        H_p: (B, T, D)
        valid_len: (B,)
        """
        B, T, D = H_p.shape
        q = self.ln_q(h_t).unsqueeze(1)      # (B, 1, D)
        kv = self.ln_kv(H_p)                 # (B, T, D)

        pad_mask = self._key_padding_mask(valid_len, T)  # (B, T), True=PAD
        tau = self.log_tau.exp().clamp(min=0.25, max=4.0)

        # MultiheadAttention scales by 1/sqrt(D). We multiply the query by tau to adjust effective temperature.
        q_scaled = q * tau

        ctx, _ = self.mha(q_scaled, kv, kv, key_padding_mask=pad_mask, need_weights=False)  # (B, 1, D)
        ctx = ctx.squeeze(1)                           # (B, D)
        # Residual + norm (post)
        out = self.ln_out(ctx + h_t)
        return out
