import torch
from ptliq.model.mv_dgt import MultiViewDGT


def _toy_masks(E):
    m_struct = torch.ones(E, dtype=torch.bool)
    m_port = torch.zeros(E, dtype=torch.bool)
    m_cg = torch.zeros(E, dtype=torch.bool)
    m_cl = torch.zeros(E, dtype=torch.bool)
    return {
        'struct': m_struct,
        'port': m_port,
        'corr_global': m_cg,
        'corr_local': m_cl,
    }


def _build_min_model(x_dim=8, hidden=16, mkt_dim=0, trade_dim=2, use_basket=True, basket_mode='residual'):
    # small 4-node graph with simple edges
    N = 6
    src = torch.tensor([0,1,2,3,4,5, 1,3,5,0])
    dst = torch.tensor([1,2,3,4,5,0, 2,4,0,3])
    edge_index = torch.stack([src, dst], dim=0)
    E = edge_index.size(1)
    masks = _toy_masks(E)
    model = MultiViewDGT(
        x_dim=x_dim,
        hidden=hidden,
        heads=2,
        dropout=0.0,
        view_masks=masks,
        edge_index=edge_index,
        edge_weight=torch.ones(E),
        mkt_dim=mkt_dim,
        use_portfolio=False,
        use_market=bool(mkt_dim > 0),
        trade_dim=trade_dim,
        view_names=['struct','port','corr_global','corr_local'],
        use_pf_head=False,
        # basket attn
        use_portfolio_attn=use_basket,
        portfolio_attn_layers=1,
        portfolio_attn_heads=2,
        portfolio_attn_dropout=0.0,
        portfolio_attn_hidden=None,
        portfolio_attn_concat_trade=True,
        portfolio_attn_concat_market=False,
        portfolio_attn_mode=basket_mode,
        portfolio_attn_gate_init=0.0,
        max_portfolio_len=None,
    )
    return model


def test_residual_basket_changes_focal_item_when_peers_change():
    torch.manual_seed(123)
    N, B = 10, 3
    x = torch.randn(N, 8)
    model = _build_min_model(use_basket=True, basket_mode='residual')

    # three items in one portfolio (gid=0)
    anchor = torch.tensor([1, 2, 3], dtype=torch.long)
    pf_gid = torch.tensor([0, 0, 0], dtype=torch.long)

    # market disabled in this test
    mkt = None

    # trade features: [side_sign, log_size]
    trade = torch.tensor([
        [ 1.0,  0.2],  # focal item idx 0
        [-1.0,  0.5],  # peer A
        [ 1.0, -0.3],  # peer B
    ], dtype=torch.float32)

    # baseline prediction with given basket
    y0 = model(x, anchor_idx=anchor, market_feat=mkt, pf_gid=pf_gid, port_ctx=None, trade_feat=trade)

    # modify peers substantially while keeping focal item (index 0) fixed
    trade_mod = trade.clone()
    trade_mod[1, 0] *= -1.0  # flip side for peer A
    trade_mod[2, 1] *= 10.0  # scale size for peer B

    y1 = model(x, anchor_idx=anchor, market_feat=mkt, pf_gid=pf_gid, port_ctx=None, trade_feat=trade_mod)

    # focal item score should change (basket residual should affect)
    assert not torch.allclose(y0[0], y1[0], atol=1e-6), "Residual basket attention had no effect on focal item"


def test_residual_vs_disabled_models_differ_for_multi_item_basket():
    torch.manual_seed(321)
    N = 10
    x = torch.randn(N, 8)
    anchor = torch.tensor([4, 5], dtype=torch.long)
    pf_gid = torch.tensor([1, 1], dtype=torch.long)
    trade = torch.tensor([[1.0, 0.1], [-1.0, 0.2]], dtype=torch.float32)

    m_basket = _build_min_model(use_basket=True, basket_mode='residual')
    m_none = _build_min_model(use_basket=False)

    y_basket = m_basket(x, anchor_idx=anchor, market_feat=None, pf_gid=pf_gid, port_ctx=None, trade_feat=trade)
    y_none = m_none(x, anchor_idx=anchor, market_feat=None, pf_gid=pf_gid, port_ctx=None, trade_feat=trade)

    # At least one of the two outputs should differ when basket attention is enabled
    assert not torch.allclose(y_basket, y_none, atol=1e-6), "Models with and without basket attention produced identical outputs for a multi-item basket" 
