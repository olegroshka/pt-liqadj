import pytest
import torch


def test_graphencoder_requires_heads_divisible():
    from ptliq.model.model import GraphEncoder
    # d_model not divisible by heads should raise
    with pytest.raises(AssertionError):
        GraphEncoder(x_dim=8, num_relations=3, d_model=130, heads=4)
    # heads must be positive
    with pytest.raises(AssertionError):
        GraphEncoder(x_dim=8, num_relations=3, d_model=128, heads=0)


def test_backbone_portfolio_index_guards_mismatch_and_oob():
    from ptliq.model.backbone import LiquidityResidualBackbone

    N, d = 10, 16
    node_embeddings = torch.randn(N, d)
    B = 3
    target_index = torch.randint(0, N, (B,))

    backbone = LiquidityResidualBackbone(d_model=d)

    # Mismatched lengths of port_index and port_batch
    port_index = torch.tensor([0, 1, 2, 3, 4])
    port_batch = torch.tensor([0, 1, 2, 0])  # one fewer
    with pytest.raises(AssertionError):
        backbone.forward_from_node_embeddings(
            node_embeddings, target_index, port_index, port_batch, None
        )

    # Out-of-bounds index in port_index
    port_index = torch.tensor([0, 1, N])  # N is out-of-range
    port_batch = torch.tensor([0, 1, 2])
    with pytest.raises(AssertionError):
        backbone.forward_from_node_embeddings(
            node_embeddings, target_index, port_index, port_batch, None
        )


def test_backbone_valid_empty_portfolio_smoke():
    from ptliq.model.backbone import LiquidityResidualBackbone

    N, d = 10, 16
    node_embeddings = torch.randn(N, d)
    B = 4
    target_index = torch.randint(0, N, (B,))
    port_index = torch.tensor([], dtype=torch.long)
    port_batch = torch.tensor([], dtype=torch.long)

    backbone = LiquidityResidualBackbone(d_model=d)
    out = backbone.forward_from_node_embeddings(
        node_embeddings, target_index, port_index, port_batch, None
    )
    assert out['delta_mean'].shape == (B, 1)
    assert out['q50'].shape == (B, 1)
    assert out['q90'].shape == (B, 1)
