import pytest
import torch


def _has_pyg():
    try:
        import torch_geometric  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _has_pyg(), reason="PyG not installed in this environment")
def test_model_outputs_are_deltas():
    from torch_geometric.data import Data
    from ptliq.model.model import LiquidityModelGAT

    torch.manual_seed(123)

    # Tiny graph
    N = 10
    x_dim = 6
    num_rel = 4
    d_model = 16

    x = torch.randn(N, x_dim)
    # Make a simple chain graph (undirected)
    src = torch.arange(N - 1)
    dst = torch.arange(1, N)
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    E = edge_index.size(1)

    edge_type = torch.randint(0, num_rel, (E,))
    edge_weight = torch.rand(E)

    data = Data(x=x, edge_index=edge_index, edge_type=edge_type, edge_weight=edge_weight, num_nodes=N)

    # Build a small batch with portfolios
    B = 3
    target_index = torch.tensor([0, 4, 8], dtype=torch.long)
    port_index = torch.tensor([1, 2, 3, 5, 6, 9], dtype=torch.long)
    port_batch = torch.tensor([0, 0, 0, 1, 1, 2], dtype=torch.long)
    port_weight = torch.randn(port_index.numel())

    # Baseline features passed to the model (size/side/log_size etc.).
    # Ensure they are not mixed into the portfolio tower (model must output deltas).
    baseline_feats = torch.randn(B, 3)

    model = LiquidityModelGAT(
        x_dim=x_dim,
        num_relations=num_rel,
        d_model=d_model,
        heads=2,
        issuer_emb_dim=0,
        encoder_type="gat_diff",
        baseline_dim=baseline_feats.size(1),
    )

    out = model.forward_from_data(
        data,
        target_index=target_index,
        port_index=port_index,
        port_batch=port_batch,
        port_weight=port_weight,
        baseline_feats=baseline_feats,
    )

    # The model must return delta outputs (not totals)
    for k in ("delta_mean", "q50", "q90"):
        assert k in out, f"Missing key {k} in model output"
        assert out[k].shape == (B, 1), f"Output {k} has wrong shape {out[k].shape}, expected {(B,1)}"
    # Non-crossing by construction
    assert torch.all(out["q90"] >= out["q50"])