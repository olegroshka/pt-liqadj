import pytest
import torch


def _has_pyg():
    try:
        import torch_geometric  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _has_pyg(), reason="PyG not installed in this environment")
def test_graph_encoder_shapes():
    from torch_geometric.data import Data
    from ptliq.model.model import GraphEncoder

    N = 10; x_dim = 7; num_rel = 3; d_model = 16
    x = torch.randn(N, x_dim)
    # simple chain graph
    src = torch.arange(N-1); dst = torch.arange(1, N)
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    E = edge_index.size(1)
    edge_type = torch.randint(0, num_rel, (E,))
    edge_weight = torch.randn(E)
    issuer_index = torch.randint(0, 4, (N,))

    enc = GraphEncoder(x_dim=x_dim, num_relations=num_rel, d_model=d_model, issuer_emb_dim=8, heads=4, rel_emb_dim=8)
    h = enc(x, edge_index, edge_type, edge_weight=edge_weight, issuer_index=issuer_index)
    assert h.shape == (N, d_model)


@pytest.mark.skipif(not _has_pyg(), reason="PyG not installed in this environment")
def test_backbone_from_node_embeddings_and_empty_port():
    from ptliq.model.backbone import LiquidityResidualBackbone
    d = 32
    B = 3
    N = 20
    node_embeddings = torch.randn(N, d)
    target_index = torch.tensor([0, 5, 7])
    # T tokens, assign none to basket 1 to test empty portfolio case
    port_index = torch.tensor([1,2,3,8,9,10,11,12])
    port_batch = torch.tensor([0,0,0, 2,2,2,2,2])
    port_weight = torch.randn(port_index.numel())

    bb = LiquidityResidualBackbone(d_model=d, n_heads=4, dropout=0.0)
    out = bb.forward_from_node_embeddings(node_embeddings, target_index, port_index, port_batch, port_weight)
    for k in ('delta_mean', 'q50', 'q90'):
        assert k in out
        assert out[k].shape == (B, 1)
    # non-crossing by construction
    assert torch.all(out['q90'] >= out['q50'])


@pytest.mark.skipif(not _has_pyg(), reason="PyG not installed in this environment")
def test_full_model_and_loss_end_to_end():
    from torch_geometric.data import Data
    from ptliq.model.model import LiquidityModelGAT
    from ptliq.model.losses import composite_loss

    torch.manual_seed(0)
    N = 15; x_dim = 6; num_rel = 4; d_model = 16
    x = torch.randn(N, x_dim)
    # simple star graph
    center = torch.zeros(N-1, dtype=torch.long)
    leaves = torch.arange(1, N)
    edge_index = torch.stack([torch.cat([center, leaves]), torch.cat([leaves, center])], dim=0)
    E = edge_index.size(1)
    edge_type = torch.randint(0, num_rel, (E,))
    edge_weight = torch.randn(E)
    issuer_index = torch.randint(0, 5, (N,))

    data = Data(x=x, edge_index=edge_index, edge_type=edge_type, edge_weight=edge_weight, issuer_index=issuer_index)

    model = LiquidityModelGAT(x_dim=x_dim, num_relations=num_rel, d_model=d_model, issuer_emb_dim=8, heads=4)

    B = 4
    target_index = torch.tensor([1, 2, 3, 4])
    port_index = torch.tensor([5,6,7, 8,9, 10,11,12, 13,14])
    port_batch = torch.tensor([0,0,0, 1,1, 2,2,2, 3,3])
    port_weight = torch.randn(port_index.numel())

    pred = model.forward_from_data(data, target_index, port_index, port_batch, port_weight)
    for k in ('delta_mean', 'q50', 'q90'):
        assert k in pred and pred[k].shape == (B, 1)
        assert torch.isfinite(pred[k]).all()

    r_true = torch.randn(B)
    lref = torch.randn(B)
    losses = composite_loss(pred, r_true, lref)
    assert 'total' in losses and torch.isfinite(losses['total'])
    # basic guardrails non-negative
    assert losses['noncross'] >= 0 and losses['wpen'] >= 0
