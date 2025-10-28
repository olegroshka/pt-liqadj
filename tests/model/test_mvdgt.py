import torch
from ptliq.model.mv_dgt import MultiViewDGT


def _toy_masks(E):
    # struct: first half; port: second quarter; corr_global: third quarter; corr_local: last quarter
    m_struct = torch.zeros(E, dtype=torch.bool); m_struct[:E//2] = True
    m_port = torch.zeros(E, dtype=torch.bool); m_port[E//2:3*E//4] = True
    m_cg = torch.zeros(E, dtype=torch.bool); m_cg[3*E//4- (E//8):3*E//4] = True if E>=8 else m_cg
    m_cl = torch.zeros(E, dtype=torch.bool); m_cl[3*E//4:] = True
    return {
        'struct': m_struct,
        'port': m_port,
        'corr_global': m_cg,
        'corr_local': m_cl,
        'corr_any': m_cg | m_cl,
    }


def test_mvdgt_forward_shapes_and_no_nan():
    torch.manual_seed(0)
    N, x_dim = 6, 5
    x = torch.randn(N, x_dim)
    # Build a small directed edge_index with 12 edges
    src = torch.tensor([0,1,2,3,4,5, 0,2,4,1,3,5])
    dst = torch.tensor([1,2,3,4,5,0, 2,3,5,0,2,4])
    edge_index = torch.stack([src, dst], dim=0)
    E = edge_index.size(1)
    edge_weight = torch.ones(E)
    masks = _toy_masks(E)

    model = MultiViewDGT(x_dim=x_dim, hidden=16, heads=2, dropout=0.0,
                         view_masks=masks, edge_index=edge_index,
                         edge_weight=edge_weight, mkt_dim=3,
                         use_portfolio=True, use_market=True)

    anchor = torch.tensor([0, 2, 4, 1], dtype=torch.long)
    mkt = torch.randn(len(anchor), 3)

    # Dummy portfolio context: one group with nodes (1,3) weights (0.6, 0.4)
    port_ctx = {
        'port_nodes_flat': torch.tensor([1,3], dtype=torch.long),
        'port_w_signed_flat': torch.tensor([0.6, 0.4], dtype=torch.float32),
        'port_offsets': torch.tensor([0], dtype=torch.long),
        'port_len': torch.tensor([2], dtype=torch.long),
    }
    pf_gid = torch.tensor([0, 0, -1, 0], dtype=torch.long)

    yhat = model(x, anchor_idx=anchor, market_feat=mkt, pf_gid=pf_gid, port_ctx=port_ctx)
    assert yhat.shape == (len(anchor),)
    assert torch.isfinite(yhat).all()


def test_portfolio_vectors_weighted_sum():
    torch.manual_seed(0)
    # craft a hidden matrix H with known rows
    H = torch.tensor([[1.,0.],[0.,1.],[1.,1.],[2.,1.]], dtype=torch.float32)
    # portfolio group with nodes [1,3] weights [0.25, 0.75]
    port_ctx = {
        'port_nodes_flat': torch.tensor([1,3], dtype=torch.long),
        'port_w_signed_flat': torch.tensor([0.25, 0.75], dtype=torch.float32),
        'port_offsets': torch.tensor([0], dtype=torch.long),
        'port_len': torch.tensor([2], dtype=torch.long),
    }
    masks = {'struct': torch.zeros(4, dtype=torch.bool), 'port': torch.zeros(4, dtype=torch.bool),
             'corr_global': torch.zeros(4, dtype=torch.bool), 'corr_local': torch.zeros(4, dtype=torch.bool)}
    # minimal model for calling helper
    model = MultiViewDGT(x_dim=2, hidden=2, heads=1, dropout=0.0,
                         view_masks=masks, edge_index=torch.zeros(2,4, dtype=torch.long))
    pf_gid = torch.tensor([0, -1], dtype=torch.long)
    out = model._portfolio_vectors(H, pf_gid, port_ctx)
    # expected: 0.25*[0,1] + 0.75*[2,1] = [1.5, 1.0]
    exp0 = torch.tensor([1.5, 1.0])
    assert torch.allclose(out[0], exp0, atol=1e-6)
    assert torch.allclose(out[1], torch.zeros(2), atol=1e-6)
