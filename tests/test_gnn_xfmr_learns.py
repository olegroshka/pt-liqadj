from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import pandas as pd

from ptliq.data.simulate import simulate, SimParams
from ptliq.data.split import compute_default_ranges, write_ranges
from ptliq.features.graph_dataset import build_graph_inputs_for_split
from ptliq.training.gnn_loop import train_gnn, GNNTrainConfig


def _inject_portfolio_dependent_targets(gi):
    """
    Build a portfolio-dependent regression target:
      y = 0.8 * fraction_of_same_sector_in_portfolio + 0.2 * coupon + noise
    where 'portfolio' is the set of trades on the same day (padded with -1).
    """
    device = gi.node_ids.device
    node_to_sector = gi.node_to_sector.to(device)  # [N] long
    node_ids = gi.node_ids.to(device)              # [B]
    port_nodes = gi.port_nodes.to(device)          # [B, T] with pad=-1
    pad_mask = port_nodes < 0
    port_nodes = torch.clamp(port_nodes, min=0)

    # sector of the target trade
    sec_target = node_to_sector[node_ids]          # [B]

    # sectors of portfolio members
    sec_port = node_to_sector[port_nodes]          # [B, T]
    same = (sec_port == sec_target.unsqueeze(1))   # [B, T] bool
    same = same.masked_fill(pad_mask, False)       # ignore padding
    valid_counts = (~pad_mask).sum(1).clamp(min=1) # [B]

    frac_same = same.float().sum(1) / valid_counts.float()   # [B]

    # Use coupon as a second numeric driver; it's the first numeric column in GraphInputs
    coupon = gi.nums[:, 0]                                   # [B]
    # normalize coupon to ~[0,1] (simple z->0..1 via min-max within batch)
    c_min = torch.min(coupon)
    c_max = torch.max(coupon)
    c_norm = (coupon - c_min) / (c_max - c_min + 1e-6)

    y = 0.8 * frac_same + 0.2 * c_norm + 0.01 * torch.randn_like(c_norm)
    gi.y = y.view(-1, 1)                                     # [B,1]
    return gi


def test_gnn_xfmr_learns(tmp_path: Path):
    # 1) Simulate 4 days (02..05)
    frames = simulate(SimParams(n_bonds=80, n_days=4, providers=["P1"], seed=123, outdir=tmp_path))
    (tmp_path / "bonds.parquet").write_bytes(frames["bonds"].to_parquet(index=False))
    (tmp_path / "trades.parquet").write_bytes(frames["trades"].to_parquet(index=False))

    # 2) train=02..03, val=04, test=05
    ranges = compute_default_ranges(tmp_path / "trades.parquet", train_end="2025-01-03", val_end="2025-01-04")
    rpath = write_ranges(ranges, tmp_path / "splits")

    # 3) Graph inputs
    gi_tr = build_graph_inputs_for_split(tmp_path, rpath, "train", max_port_items=64)
    gi_va = build_graph_inputs_for_split(tmp_path, rpath, "val", max_port_items=64)
    assert gi_tr.node_ids.numel() > 0 and gi_va.node_ids.numel() > 0

    gi_tr = _inject_portfolio_dependent_targets(gi_tr)
    gi_va = _inject_portfolio_dependent_targets(gi_va)

    # 4) Train GNN+Transformer (GPU if available)
    outdir = tmp_path / "models_gnn"
    cfg = GNNTrainConfig(
        device="auto",
        max_epochs=20,
        batch_size=256,
        lr=1e-3,
        patience=5,
        n_layers=2,
        nhead=4,
        d_model=128,
        gnn_out_dim=128,
        gnn_num_hidden=64,
        use_baseline=False,   # important: our synthetic rule ignores size/urgency
    )
    res = train_gnn(gi_tr, gi_va, outdir, cfg)
    assert (outdir / "ckpt_gnn.pt").exists()
    assert res["best_epoch"] > 0

    # 5) Quick validation MAE on val split
    from ptliq.model import PortfolioResidualModel, ModelConfig, NodeFieldSpec
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(outdir / "ckpt_gnn.pt", map_location="cpu")

    cat_specs = [
        NodeFieldSpec("sector_code", int(gi_tr.node_to_sector.max().item() + 1), 16),
        NodeFieldSpec("rating_code", int(gi_tr.cats["rating_code"].max().item() + 1), 12),
    ]
    mcfg = ModelConfig(
        n_nodes=gi_tr.n_nodes,
        node_id_dim=32,
        cat_specs=cat_specs,
        n_num=int(gi_tr.nums.shape[1]),
        gnn_num_hidden=64,
        gnn_out_dim=128,
        d_model=128,
        nhead=4,
        n_layers=2,
        device="auto",
        use_baseline=False,
    )
    model = PortfolioResidualModel(mcfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(dev).eval()

    with torch.no_grad():
        B = int(gi_va.node_ids.shape[0])
        bs = 256
        preds, trues = [], []
        for i in range(0, B, bs):
            sl = slice(i, i + bs)
            out = model(
                gi_va.node_ids[sl].to(dev),
                {"sector_code": gi_va.cats["sector_code"][sl].to(dev),
                 "rating_code": gi_va.cats["rating_code"][sl].to(dev)},
                gi_va.nums[sl].to(dev),
                {k: v.to(dev) for k, v in gi_va.issuer_groups.items()},
                {k: v.to(dev) for k, v in gi_va.sector_groups.items()},
                gi_va.node_to_issuer.to(dev),
                gi_va.node_to_sector.to(dev),
                gi_va.port_nodes[sl].to(dev),
                gi_va.port_len[sl].to(dev),
                None,
                None,
            )
            preds.append(out["mean"].detach().cpu())
            trues.append(gi_va.y[sl].cpu())
        pred = torch.cat(preds, dim=0).squeeze(-1)
        true = torch.cat(trues, dim=0).squeeze(-1)
        mae = torch.mean(torch.abs(pred - true)).item()

    assert mae < 0.10, f"validation MAE too high: {mae:.3f}"
