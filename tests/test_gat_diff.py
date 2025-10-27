import pytest
import torch


def _has_pyg():
    try:
        import torch_geometric  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _has_pyg(), reason="PyG not installed in this environment")
def test_liquidity_model_gat_diff_forward_with_corr_mask():
    from torch_geometric.data import Data
    from ptliq.model.model import LiquidityModelGAT

    torch.manual_seed(42)
    # Small graph
    N = 12
    x_dim = 5
    num_rel = 6
    d_model = 16

    x = torch.randn(N, x_dim)
    # Build a simple ring graph
    src = torch.arange(N)
    dst = (src + 1) % N
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    E = edge_index.size(1)

    # Random relation ids and weights
    edge_type = torch.randint(0, num_rel, (E,))
    edge_weight = torch.rand(E)
    issuer_index = torch.randint(0, 4, (N,))

    # Mark half the edges as correlation edges via mask
    corr_mask = torch.zeros(E, dtype=torch.bool)
    corr_mask[: E // 2] = True

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_weight=edge_weight,
        num_nodes=N,
    )
    data.issuer_index = issuer_index
    data.corr_edge_mask = corr_mask

    # Build a simple portfolio batch
    B = 3
    target_index = torch.tensor([0, 4, 8], dtype=torch.long)
    port_index = torch.tensor([1, 2, 3, 5, 6, 9, 10, 11], dtype=torch.long)
    port_batch = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2], dtype=torch.long)
    port_weight = torch.rand(port_index.numel())

    model = LiquidityModelGAT(
        x_dim=x_dim,
        num_relations=num_rel,
        d_model=d_model,
        issuer_emb_dim=8,
        heads=4,
        rel_emb_dim=8,
        encoder_type="gat_diff",
    )

    out = model.forward_from_data(data, target_index, port_index, port_batch, port_weight)
    # Basic shape and finiteness checks
    for k in ("delta_mean", "q50", "q90"):
        assert k in out and out[k].shape == (B, 1)
        assert torch.isfinite(out[k]).all()
    # Non-crossing quantiles
    assert torch.all(out["q90"] >= out["q50"])  # by model head construction


@pytest.mark.skipif(not _has_pyg(), reason="PyG not installed in this environment")
def test_liquidity_model_gat_diff_forward_without_mask_graceful():
    # If no corr_edge_mask is present, gat_diff should behave like base path (no refinement)
    from torch_geometric.data import Data
    from ptliq.model.model import LiquidityModelGAT

    torch.manual_seed(0)
    N = 8; x_dim = 4; num_rel = 3; d_model = 16
    x = torch.randn(N, x_dim)
    src = torch.arange(N - 1)
    dst = torch.arange(1, N)
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    E = edge_index.size(1)
    edge_type = torch.randint(0, num_rel, (E,))
    edge_weight = torch.rand(E)

    data = Data(x=x, edge_index=edge_index, edge_type=edge_type, edge_weight=edge_weight, num_nodes=N)

    target_index = torch.tensor([0, 2], dtype=torch.long)
    port_index = torch.tensor([3, 4, 5, 6, 7], dtype=torch.long)
    port_batch = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    port_weight = torch.ones(port_index.numel())

    model = LiquidityModelGAT(x_dim=x_dim, num_relations=num_rel, d_model=d_model, heads=4, issuer_emb_dim=0, encoder_type="gat_diff")
    out = model.forward_from_data(data, target_index, port_index, port_batch, port_weight)

    for k in ("delta_mean", "q50", "q90"):
        assert k in out and out[k].shape == (2, 1)
        assert torch.isfinite(out[k]).all()
