# ptliq/training/gnn_loop.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple

import json
import math
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW

from ptliq.model import (
    PortfolioResidualModel,
    ModelConfig,
    NodeFieldSpec,
    resolve_device,
)
from ptliq.model.utils import GraphInputs


@dataclass
class GNNTrainConfig:
    device: str = "auto"
    max_epochs: int = 20
    batch_size: int = 256
    lr: float = 1e-3
    patience: int = 5
    seed: int = 42

    # model knobs (kept aligned with tests)
    node_id_dim: int = 32
    nhead: int = 4
    n_layers: int = 2
    d_model: int = 128
    gnn_num_hidden: int = 64
    gnn_out_dim: int = 128
    gnn_dropout: float = 0.0
    head_hidden: int = 128
    head_dropout: float = 0.0
    use_calibrator: bool = False
    use_baseline: bool = False


def _iter_batches(n: int, bs: int) -> Iterator[slice]:
    if n <= 0:
        return
    for i in range(0, n, bs):
        yield slice(i, min(i + bs, n))


def _to_dev(x, dev: torch.device):
    """Recursively move tensors in nested structures to device."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.to(dev)
    if isinstance(x, dict):
        return {k: _to_dev(v, dev) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = [_to_dev(v, dev) for v in x]
        return type(x)(t) if not isinstance(x, tuple) else tuple(t)
    return x


def _pack_batch(gi: GraphInputs, sl: slice) -> Dict[str, torch.Tensor]:
    """
    Build a batch dict from GraphInputs. Handles optional fields gracefully.
    Required: node_ids, cats, nums, issuer_groups, sector_groups, node_to_issuer,
              node_to_sector, port_nodes, port_len, y.
    Optional: size_side_urg.
    """
    batch = {
        "node_ids": gi.node_ids[sl],
        "cats": {k: v[sl] for k, v in gi.cats.items()},
        "nums": gi.nums[sl],
        "issuer_groups": gi.issuer_groups,  # dict[int -> Tensor[?]] (kept global)
        "sector_groups": gi.sector_groups,  # dict[int -> Tensor[?]] (kept global)
        "node_to_issuer": gi.node_to_issuer,
        "node_to_sector": gi.node_to_sector,
        "port_nodes": gi.port_nodes[sl],
        "port_len": gi.port_len[sl],
        "y": gi.y[sl],
    }
    # Optional auxiliary features for the trade itself (size/side/urgency)
    if hasattr(gi, "size_side_urg") and gi.size_side_urg is not None:
        batch["size_side_urg"] = gi.size_side_urg[sl]
    else:
        batch["size_side_urg"] = None
    return batch


def _mk_model(gi: GraphInputs, cfg: GNNTrainConfig) -> PortfolioResidualModel:
    cat_specs = [
        NodeFieldSpec("sector_code", cardinality=int(gi.node_to_sector.max().item() + 1), emb_dim=16),
        NodeFieldSpec("rating_code", cardinality=int(gi.cats["rating_code"].max().item() + 1), emb_dim=12),
    ]
    mcfg = ModelConfig(
        n_nodes=gi.n_nodes,
        node_id_dim=cfg.node_id_dim,
        cat_specs=cat_specs,
        n_num=int(gi.nums.shape[1]),
        gnn_num_hidden=cfg.gnn_num_hidden,
        gnn_out_dim=cfg.gnn_out_dim,
        gnn_dropout=cfg.gnn_dropout,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        n_layers=cfg.n_layers,
        tr_dropout=0.0,
        head_hidden=cfg.head_hidden,
        head_dropout=cfg.head_dropout,
        use_calibrator=cfg.use_calibrator,
        use_baseline=cfg.use_baseline,
        device=cfg.device,
    )
    return PortfolioResidualModel(mcfg)


def _step(model: PortfolioResidualModel, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    out = model(
        batch["node_ids"],
        batch["cats"],
        batch["nums"],
        batch["issuer_groups"],
        batch["sector_groups"],
        batch["node_to_issuer"],
        batch["node_to_sector"],
        batch["port_nodes"],
        batch["port_len"],
        batch.get("size_side_urg", None),
        None,  # no extra mask
    )
    return model.compute_loss(out, batch["y"])


def train_gnn(
    train_gi: GraphInputs,
    val_gi: GraphInputs,
    outdir: Path,
    cfg: GNNTrainConfig,
) -> Dict[str, float]:
    """
    Train GNN+Transformer with early stopping on validation MAE.

    Returns: {"best_epoch": int, "best_val_mae_bps": float}
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dev = resolve_device(cfg.device)
    model = _mk_model(train_gi, cfg).to(dev)
    opt = AdamW(model.parameters(), lr=cfg.lr)

    best_val = float("inf")
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        tr_losses = []
        n = int(train_gi.node_ids.shape[0])
        for sl in _iter_batches(n, cfg.batch_size):
            batch = _to_dev(_pack_batch(train_gi, sl), dev)
            opt.zero_grad()
            loss = _step(model, batch)
            loss.backward()
            opt.step()
            tr_losses.append(loss.detach().item())

        # ---- validation
        model.eval()
        with torch.no_grad():
            preds, trues = [], []
            m = int(val_gi.node_ids.shape[0])
            for sl in _iter_batches(m, cfg.batch_size):
                batch = _to_dev(_pack_batch(val_gi, sl), dev)
                out = model(
                    batch["node_ids"], batch["cats"], batch["nums"],
                    batch["issuer_groups"], batch["sector_groups"],
                    batch["node_to_issuer"], batch["node_to_sector"],
                    batch["port_nodes"], batch["port_len"],
                    batch.get("size_side_urg", None), None,
                )
                preds.append(out["mean"].detach().cpu())
                trues.append(batch["y"].detach().cpu())
            pred = torch.cat(preds, 0).squeeze(-1)
            true = torch.cat(trues, 0).squeeze(-1)
            val_mae = float(torch.mean(torch.abs(pred - true)).item())

        # early stopping
        if val_mae < best_val - 1e-6:
            best_val = val_mae
            best_epoch = epoch
            # save checkpoint
            ckpt = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "val_mae_bps": best_val,
            }
            torch.save(ckpt, outdir / "ckpt_gnn.pt")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Optional: write a tiny training log (helps debugging)
        (outdir / "progress.jsonl").open("a").write(
            json.dumps({"epoch": epoch, "train_loss": float(np.mean(tr_losses)), "val_mae_bps": val_mae}) + "\n"
        )

        if epochs_no_improve >= cfg.patience:
            break

    return {"best_epoch": best_epoch, "best_val_mae_bps": best_val}
