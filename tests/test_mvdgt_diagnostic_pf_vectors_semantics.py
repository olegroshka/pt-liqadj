# tests/test_mvdgt_diagnostic_pf_vectors_semantics.py
import math
import torch
import pytest

from ptliq.model.mv_dgt import compute_samplewise_portfolio_vectors_loo

@pytest.mark.unit
def test_diag_pf_vectors_basic_weighted_sum_and_loo_prints(capfd):
    """
    Pure tensor-level check (no model). Shows:
      - inclusive absolute weighted sum direction (what you expect intuitively),
      - strict-LOO absolute mean used by the model for the anchor,
      - current V_sgn from the code,
      - factorized (expected) V_sgn = (sum_signed / sum_abs) * V_abs.
    """
    # H rows are easy to read: e0=(1,0), e1=(0,1), h2=(1,1), h3=(2,1)
    H = torch.tensor([[1.,0.],[0.,1.],[1.,1.],[2.,1.]], dtype=torch.float32)

    # Anchor = node 2 (h2); co-items in the portfolio group = [1,3]
    anchor_idx = torch.tensor([2], dtype=torch.long)
    pf_gid     = torch.tensor([0], dtype=torch.long)

    # Absolute and signed weights for the two co-items (1, 3)
    w_abs = torch.tensor([0.25, 0.75], dtype=torch.float32)
    w_sgn = torch.tensor([+0.25, +0.75], dtype=torch.float32)  # both buy, just to print

    port_ctx = {
        "port_nodes_flat": torch.tensor([1, 3], dtype=torch.long),
        "port_w_signed_flat": w_sgn.clone(),
        "port_w_abs_flat": w_abs.clone(),
        "port_len": torch.tensor([2], dtype=torch.long),
    }

    V_abs, V_sgn = compute_samplewise_portfolio_vectors_loo(
        H, anchor_idx, pf_gid, port_ctx, l2_normalize=False
    )

    # Inclusive absolute weighted sum (sanity print): 0.25*[0,1] + 0.75*[2,1] = [1.5, 1.0]
    inclusive = (w_abs[0] * H[1]) + (w_abs[1] * H[3])

    # LOO absolute prototype (others mean): since anchor (2) is not in port_ctx -> equals inclusive / sum_abs
    sum_abs = float(w_abs.sum().item())
    loo_abs_raw = inclusive / max(1e-8, sum_abs)

    # Factorized signed vector expected by "signed mass times direction"
    sum_sgn = float(w_sgn.sum().item())           # here = 1.0
    factor_ratio = (sum_sgn / max(1e-8, sum_abs)) # here = 1.0
    vsgn_factorized = factor_ratio * loo_abs_raw

    print(f"[DIAG] inclusive_abs_raw   = {inclusive.tolist()} (expect [1.5, 1.0])")
    print(f"[DIAG] LOO_abs_raw         = {loo_abs_raw.tolist()}")
    print(f"[DIAG] current_V_abs[0]    = {V_abs[0].tolist()}")
    print(f"[DIAG] current_V_sgn[0]    = {V_sgn[0].tolist()}")
    print(f"[DIAG] factorized_V_sgn[0] = {vsgn_factorized.tolist()}")

    # These should match closely (pure averaging math).
    assert torch.allclose(V_abs[0], loo_abs_raw, atol=1e-7)

@pytest.mark.unit
def test_diag_signed_cancellation_should_zero_with_factorized_semantics(capfd):
    """
    Equal and opposite signed weights -> factorized signed mass is zero => V_sgn must be near zero
    if the implementation is factorized. We print both current and expected.
    """
    H = torch.tensor([[1.,0.],[0.,1.],[1.,1.]], dtype=torch.float32)
    anchor_idx = torch.tensor([2], dtype=torch.long)  # anchor not in group
    pf_gid     = torch.tensor([0], dtype=torch.long)

    a = 0.6
    w_abs = torch.tensor([a, a], dtype=torch.float32)
    w_sgn = torch.tensor([+a, -a], dtype=torch.float32)  # equal & opposite

    port_ctx = {
        "port_nodes_flat": torch.tensor([0, 1], dtype=torch.long),
        "port_w_signed_flat": w_sgn.clone(),
        "port_w_abs_flat": w_abs.clone(),
        "port_len": torch.tensor([2], dtype=torch.long),
    }

    V_abs, V_sgn = compute_samplewise_portfolio_vectors_loo(
        H, anchor_idx, pf_gid, port_ctx, l2_normalize=False
    )

    # factorized expectation: (sum signed / sum abs) * V_abs = 0 * V_abs = [0,0]
    sum_abs = float(w_abs.sum().item())
    sum_sgn = float(w_sgn.sum().item())  # 0.0
    exp = (sum_sgn / max(1e-8, sum_abs)) * V_abs[0]

    print(f"[DIAG] V_abs[0]           = {V_abs[0].tolist()}")
    print(f"[DIAG] current V_sgn[0]   = {V_sgn[0].tolist()}")
    print(f"[DIAG] expected V_sgn[0]  = {exp.tolist()} (factorized)")

    # This assert will FAIL under non-factorized implementation (current code prints ~[0.5,-0.5]).
    assert torch.allclose(V_sgn[0], exp, atol=1e-7), \
        "Your V_sgn is not factorized -> equal and opposite signs do not cancel to zero."

@pytest.mark.unit
def test_diag_self_inclusion_is_neutral_for_anchor(capfd):
    """
    If the anchor is present in port_ctx, strict LOO must remove its contribution.
    This test demonstrates the invariance numerically with small H.
    """
    H = torch.tensor([[1.,0.],[0.,1.],[1.,1.],[2.,1.]], dtype=torch.float32)
    anchor_idx = torch.tensor([3], dtype=torch.long)  # anchor = node 3 (2,1)
    pf_gid     = torch.tensor([0], dtype=torch.long)

    # co-weights proportions we want under LOO:
    co_loo = torch.tensor([0.6, 0.4], dtype=torch.float32)

    # SELF context with any nonzero anchor, but co-weights proportional to 0.6:0.4
    port_ctx_self = {
        "port_nodes_flat": torch.tensor([3, 0, 1], dtype=torch.long),
        "port_w_abs_flat": torch.tensor([0.2, 0.48, 0.32], dtype=torch.float32),
        # sum others = 0.8 → normalized (0.6, 0.4)
        "port_w_signed_flat": torch.tensor([0.2, 0.48, 0.32], dtype=torch.float32),
        "port_len": torch.tensor([3], dtype=torch.long),
    }

    # LOO context with the same co-weights as above after normalization
    port_ctx_loo = {
        "port_nodes_flat": torch.tensor([0, 1], dtype=torch.long),
        "port_w_abs_flat": torch.tensor([0.6, 0.4], dtype=torch.float32),
        "port_w_signed_flat": torch.tensor([0.6, 0.4], dtype=torch.float32),
        "port_len": torch.tensor([2], dtype=torch.long),
    }

    V_abs_loo, V_sgn_loo = compute_samplewise_portfolio_vectors_loo(
        H, anchor_idx, pf_gid, port_ctx_loo, l2_normalize=False
    )

    V_abs_self, V_sgn_self = compute_samplewise_portfolio_vectors_loo(
        H, anchor_idx, pf_gid, port_ctx_self, l2_normalize=False
    )

    print(f"[DIAG] V_abs[LOO]  = {V_abs_loo[0].tolist()}   V_abs[SELF]  = {V_abs_self[0].tolist()}")
    print(f"[DIAG] V_sgn[LOO]  = {V_sgn_loo[0].tolist()}   V_sgn[SELF]  = {V_sgn_self[0].tolist()}")

    # They should match tightly if the anchor is truly removed on SELF path.
    assert torch.allclose(V_abs_loo[0], V_abs_self[0], atol=1e-7)
    assert torch.allclose(V_sgn_loo[0], V_sgn_self[0], atol=1e-7)
