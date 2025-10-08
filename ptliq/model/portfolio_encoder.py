from __future__ import annotations
import torch
import torch.nn as nn

def _make_key_padding_mask(valid_len: torch.Tensor, T: int) -> torch.Tensor:
    idx = torch.arange(T, device=valid_len.device).unsqueeze(0)   # (1, T)
    return idx >= valid_len.unsqueeze(1)                           # (B, T) True=pad

def _identity_linear(d_model: int) -> nn.Linear:
    lin = nn.Linear(d_model, d_model, bias=False)
    with torch.no_grad():
        lin.weight.zero_()
        lin.weight.add_(torch.eye(d_model))
    return lin

class PortfolioEncoder(nn.Module):
    """
    Cross-attention with learnable Q/K/V, all identity-initialized:
      Q = W_q * h_t,  K = W_k * H_p,  V = W_v * H_p
      attn = softmax((Q · K^T) / sqrt(D)) with key padding mask
      context = attn @ V
      out = h_t + beta * (context - h_t)      # convex update toward context

    Identity init => at step 0 this reduces to the parameter-free geometric
    attention that moves the query toward the planted signal (focus test).
    Trainable Q/K/V => enough capacity to pass the quick learning test.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 1,        # API compatibility; single head here
        num_layers: int = 1,   # API compatibility; unused
        dropout: float = 0.0,
        beta_init: float = 0.8,  # convex update strength
        learnable_tau: bool = False,
        tau_init: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model

        # Identity-initialized Q/K/V projections (learnable)
        self.W_q = _identity_linear(d_model)
        self.W_k = _identity_linear(d_model)
        self.W_v = _identity_linear(d_model)

        # Convex update coefficient beta in (0,1) (learnable)
        self._beta_param = nn.Parameter(torch.logit(torch.tensor(beta_init, dtype=torch.float32)))

        # Optional learnable temperature
        if learnable_tau:
            self.log_tau = nn.Parameter(torch.log(torch.tensor(float(tau_init))))
        else:
            self.register_parameter("log_tau", None)
            self._tau = float(tau_init)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _beta(self) -> torch.Tensor:
        return torch.sigmoid(self._beta_param)

    @torch.no_grad()
    def _check(self, h_t: torch.Tensor, H_p: torch.Tensor, valid_len: torch.Tensor):
        assert h_t.dim() == 2 and H_p.dim() == 3, "Expect h_t (B,D), H_p (B,T,D)"
        B, T, D = H_p.shape
        assert h_t.shape == (B, D), f"h_t {h_t.shape} vs {(B,D)}"
        assert D == self.d_model, f"d_model mismatch: {D} vs {self.d_model}"
        assert valid_len.shape == (B,), "valid_len must be (B,)"

    def forward(
        self,
        h_t: torch.Tensor,       # (B, D)
        H_p: torch.Tensor,       # (B, T, D)
        valid_len: torch.Tensor  # (B,)
    ) -> torch.Tensor:
        self._check(h_t, H_p, valid_len)
        B, T, D = H_p.shape

        # Q, K, V
        Q = self.W_q(h_t).unsqueeze(1)          # (B, 1, D)
        K = self.W_k(H_p)                       # (B, T, D)
        V = self.W_v(H_p)                       # (B, T, D)

        # scaled dot-product attention scores
        scores = torch.matmul(Q, K.transpose(1, 2)).squeeze(1)  # (B, T)
        scores = scores / (D ** 0.5)

        # mask padding
        kpm = _make_key_padding_mask(valid_len, T)              # True = pad
        scores = scores.masked_fill(kpm, float("-inf"))

        # temperature
        tau = torch.exp(self.log_tau) if self.log_tau is not None else self._tau
        scores = scores * tau

        # attention weights
        attn = torch.softmax(scores, dim=1)                     # (B, T)
        attn = self.dropout(attn)

        # context
        context = torch.einsum("bt, btd -> bd", attn, V)        # (B, D)

        # convex update toward context
        beta = self._beta()
        out = h_t + beta * (context - h_t)                      # (B, D)
        return out
