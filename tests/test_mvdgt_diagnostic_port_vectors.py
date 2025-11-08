import math
import numpy as np
import pytest
import torch

from ptliq.model.mv_dgt import (
    compute_samplewise_portfolio_vectors_loo,
    MultiViewDGT,
)

def _mk_empty_masks(E: int = 0):
    # All-zero masks of length E (works with E=0)
    z = torch.zeros(E, dtype=torch.bool)
    return {"struct": z.clone(), "port": z.clone(), "corr_global": z.clone(), "corr_local": z.clone()}

def _inclusive_weighted_sum(H, nodes_flat, w_abs_flat, group_lens, nid_anchor, l2_normalize=False):
    """
    Baseline *inclusive* prototype used for sanity checks in unit-level math.
    This is NOT what LOO should do; it includes anchor if present.
    """
    dev, dt = H.device, H.dtype
    gid = torch.repeat_interleave(torch.arange(len(group_lens), device=dev), group_lens)
    H_lines = H.index_select(0, nodes_flat)
    S = torch.zeros(len(group_lens), H.size(1), device=dev, dtype=dt)
    S.index_add_(0, gid, w_abs_flat.unsqueeze(1) * H_lines)
    # assume single group for this helper
    v = S[0]
    if l2_normalize:
        v = torch.nn.functional.normalize(v, p=2, dim=0, eps=1e-6)
    return v

@pytest.mark.unit
def test_diag_pf_vectors_math_and_semantics(capfd):
    """
    Purpose: expose differences between (a) inclusive prototype vs (b) strict LOO,
    and between raw weighted average vs l2 normalized outputs.
    """
    torch.manual_seed(0)
    # Hidden matrix with known rows:
    # 0:[1,0], 1:[0,1], 2:[1,1], 3:[2,1]
    H = torch.tensor([[1.,0.],[0.,1.],[1.,1.],[2.,1.]], dtype=torch.float32)

    # single portfolio group with nodes [1,3], w_abs=[0.25, 0.75]; anchor at node 1
    port_ctx = {
        "port_nodes_flat": torch.tensor([1, 3], dtype=torch.long),
        "port_w_signed_flat": torch.tensor([0.25, 0.75], dtype=torch.float32),
        "port_len": torch.tensor([2], dtype=torch.long),
    }
    device = H.device
    w_abs = port_ctx["port_w_signed_flat"].abs().to(device)
    nodes = port_ctx["port_nodes_flat"].to(device)
    lens  = port_ctx["port_len"].to(device)

    # (A) Inclusive raw sum baseline (what a non-LOO, raw test expects)
    v_incl_raw = _inclusive_weighted_sum(H, nodes, w_abs, lens, nid_anchor=1, l2_normalize=False)
    # Expected inclusive raw: 0.25*[0,1] + 0.75*[2,1] = [1.5, 1.0]
    exp_incl = torch.tensor([1.5, 1.0], dtype=torch.float32)
    print(f"[DIAG] inclusive_raw = {v_incl_raw.tolist()}  | expected = {exp_incl.tolist()}")

    # (B) Strict LOO: anchor=1 -> excludes its own contribution; remaining mass=0.75*H[3]; avg -> H[3]
    anchor_idx = torch.tensor([1], dtype=torch.long)
    pf_gid     = torch.tensor([0], dtype=torch.long)
    V_abs_raw, V_sgn_raw = compute_samplewise_portfolio_vectors_loo(
        H, anchor_idx, pf_gid, port_ctx, l2_normalize=False
    )
    print(f"[DIAG] LOO_abs_raw   = {V_abs_raw[0].tolist()} (should equal H[3]=[2,1])")
    print(f"[DIAG] LOO_sgn_raw   = {V_sgn_raw[0].tolist()}")

    # (C) With l2_normalize=True, you should see normalized directions
    V_abs_n, V_sgn_n = compute_samplewise_portfolio_vectors_loo(
        H, anchor_idx, pf_gid, port_ctx, l2_normalize=True
    )
    print(f"[DIAG] LOO_abs_norm  = {V_abs_n[0].tolist()}  | ||.||={float(V_abs_n[0].norm().item()):.6f}")

    # Assertions are loose: we want visibility more than brittleness here.
    assert torch.allclose(v_incl_raw, exp_incl, atol=1e-6)
    assert torch.allclose(V_abs_raw[0], H[3], atol=1e-6)  # confirms true LOO -> average of *others*
    # normalization check
    assert abs(float(V_abs_n[0].norm().item()) - 1.0) < 1e-5

@pytest.mark.unit
def test_diag_signed_cancellation_in_vectors(capfd):
    """
    Equal and opposite signed weights for co-items should make V_sgn ~ 0, while V_abs stays > 0.
    """
    H = torch.tensor([[1.,0.],[0.,1.],[1.,1.]], dtype=torch.float32)
    # group nodes [0,1] with +a and -a; anchor=2 not in group (tests "anchor not present" path)
    a = 0.6
    port_ctx = {
        "port_nodes_flat": torch.tensor([0, 1], dtype=torch.long),
        "port_w_signed_flat": torch.tensor([+a, -a], dtype=torch.float32),
        "port_len": torch.tensor([2], dtype=torch.long),
    }
    anchor_idx = torch.tensor([2], dtype=torch.long)
    pf_gid     = torch.tensor([0], dtype=torch.long)
    V_abs, V_sgn = compute_samplewise_portfolio_vectors_loo(H, anchor_idx, pf_gid, port_ctx, l2_normalize=False)
    print(f"[DIAG] signed-cancel: V_abs={V_abs[0].tolist()} V_sgn={V_sgn[0].tolist()}")
    assert torch.allclose(V_sgn[0], torch.zeros_like(V_sgn[0]), atol=1e-7)
    assert V_abs[0].norm().item() > 0.0
