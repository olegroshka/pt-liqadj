import torch
from ptliq.model.mv_dgt import MultiViewDGT


def _toy_graph(N: int):
    # simple ring graph for structural edges; duplicate edges for directionality
    src = torch.arange(N)
    dst = (src + 1) % N
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    E = edge_index.size(1)
    edge_weight = torch.ones(E)
    # masks: all edges are structural; others empty
    masks = {
        'struct': torch.ones(E, dtype=torch.bool),
        'port': torch.zeros(E, dtype=torch.bool),
        'corr_global': torch.zeros(E, dtype=torch.bool),
        'corr_local': torch.zeros(E, dtype=torch.bool),
    }
    return edge_index, edge_weight, masks


def _make_model(N: int, x_dim: int = 4, use_basket_attn: bool = True):
    edge_index, edge_weight, masks = _toy_graph(N)
    model = MultiViewDGT(
        x_dim=x_dim,
        hidden=16,
        heads=2,
        dropout=0.0,
        view_masks=masks,
        edge_index=edge_index,
        edge_weight=edge_weight,
        mkt_dim=0,
        use_portfolio=True,
        use_market=False,
        trade_dim=0,
        use_pf_head=False,
        use_portfolio_attn=use_basket_attn,
        portfolio_attn_layers=1,
        portfolio_attn_heads=2,
        portfolio_attn_dropout=0.0,
        portfolio_attn_mode='residual',
    )
    return model


def _rand_inputs(N: int, B: int, G_lens: list[int]):
    torch.manual_seed(0)
    x = torch.randn(N, 4)
    # anchors are first sum(G_lens) nodes; ensure disjoint across groups sequentially
    anchor = torch.tensor([i for i in range(sum(G_lens))], dtype=torch.long)
    # build pf_gid per item
    gids = []
    for g, L in enumerate(G_lens):
        gids.extend([g] * L)
    pf_gid = torch.tensor(gids, dtype=torch.long)
    # portfolio context matching those items; weights 1.0 for simplicity
    port_nodes = anchor.clone()
    port_w_signed = torch.ones(len(port_nodes), dtype=torch.float32)
    port_len = torch.tensor(G_lens, dtype=torch.long)
    port_ctx = {
        'port_nodes_flat': port_nodes,
        'port_w_signed_flat': port_w_signed,
        'port_len': port_len,
    }
    return x, anchor, pf_gid, port_ctx


def test_permutation_invariance_within_portfolio():
    torch.manual_seed(0)
    N = 20
    # two groups of sizes 3 and 4
    G_lens = [3, 4]
    B = sum(G_lens)
    model = _make_model(N, use_basket_attn=True)
    x, anchor, pf_gid, port_ctx = _rand_inputs(N, B, G_lens)

    # baseline order
    y0 = model(x, anchor_idx=anchor, pf_gid=pf_gid, port_ctx=port_ctx).detach()

    # permute items within each group independently
    idx_group0 = torch.arange(0, G_lens[0])
    idx_group1 = torch.arange(G_lens[0], G_lens[0] + G_lens[1])
    perm0 = idx_group0[torch.randperm(G_lens[0])]
    perm1 = idx_group1[torch.randperm(G_lens[1])]
    perm = torch.cat([perm0, perm1], dim=0)

    y1 = model(x, anchor_idx=anchor.index_select(0, perm), pf_gid=pf_gid.index_select(0, perm), port_ctx={
        'port_nodes_flat': port_ctx['port_nodes_flat'].index_select(0, perm),
        'port_w_signed_flat': port_ctx['port_w_signed_flat'].index_select(0, perm),
        'port_len': port_ctx['port_len'],
    }).detach()

    # restore y1 to original order
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(B)
    y1_restored = y1.index_select(0, inv)

    assert torch.allclose(y0, y1_restored, atol=1e-6), "Scores must be invariant to within-portfolio permutation"


def test_cross_portfolio_isolation():
    torch.manual_seed(1)
    N = 30
    G_lens = [3, 5]
    B = sum(G_lens)
    model = _make_model(N, use_basket_attn=True)
    x, anchor, pf_gid, port_ctx = _rand_inputs(N, B, G_lens)

    # score baseline
    y_base = model(x, anchor_idx=anchor, pf_gid=pf_gid, port_ctx=port_ctx).detach()

    # modify portfolio 2 (group index 1) by changing its weights and even swapping two nodes beyond max_portfolio_len concerns
    port_nodes2 = port_ctx['port_nodes_flat'].clone()
    start = G_lens[0]
    end = G_lens[0] + G_lens[1]
    # reverse the second group's order and double weights
    port_nodes2[start:end] = torch.flip(port_nodes2[start:end], dims=[0])
    port_w2 = port_ctx['port_w_signed_flat'].clone()
    port_w2[start:end] = 2.0
    port_ctx2 = {
        'port_nodes_flat': port_nodes2,
        'port_w_signed_flat': port_w2,
        'port_len': port_ctx['port_len'],
    }
    y_mod = model(x, anchor_idx=anchor, pf_gid=pf_gid, port_ctx=port_ctx2).detach()

    # first group's indices
    g0_idx = torch.arange(0, G_lens[0])
    assert torch.allclose(y_base.index_select(0, g0_idx), y_mod.index_select(0, g0_idx), atol=1e-6), \
        "Changing PF#2 must not affect PF#1 scores in the same batch"


def test_singleton_stability_matches_disabled_attention():
    torch.manual_seed(2)
    N = 10
    G_lens = [1, 1, 1, 1]
    B = sum(G_lens)
    x, anchor, pf_gid, port_ctx = _rand_inputs(N, B, G_lens)

    model_off = _make_model(N, use_basket_attn=False)
    model_on = _make_model(N, use_basket_attn=True)
    # Align shared weights to isolate basket encoder effect
    model_on.load_state_dict(model_off.state_dict(), strict=False)

    y_off = model_off(x, anchor_idx=anchor, pf_gid=pf_gid, port_ctx=port_ctx).detach()
    y_on = model_on(x, anchor_idx=anchor, pf_gid=pf_gid, port_ctx=port_ctx).detach()

    assert torch.allclose(y_off, y_on, atol=1e-6), "Singleton portfolios should behave identically with or without basket attention"


def test_backward_compat_off_runs_and_shapes():
    torch.manual_seed(3)
    N = 12
    model = _make_model(N, use_basket_attn=False)
    x = torch.randn(N, 4)
    anchor = torch.tensor([0, 3, 5, 7], dtype=torch.long)
    # no portfolio info provided
    y = model(x, anchor_idx=anchor)
    assert y.shape == (anchor.size(0),)
    assert torch.isfinite(y).all()
