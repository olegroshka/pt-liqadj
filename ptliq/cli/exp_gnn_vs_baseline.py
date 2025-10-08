# ptliq/cli/exp_gnn_vs_baseline.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import typer
from rich import print

from ptliq.utils.config import load_config, RootConfig, get_sim_config
from ptliq.data.simulate import simulate, SimParams
from ptliq.data.split import compute_default_ranges, write_ranges

# Graph builders + synthetic target injector (names vary slightly across branches)
try:
    from ptliq.model.utils import (
        build_graph_inputs_for_split,
        _inject_portfolio_dependent_targets as inject_targets,
    )
except ImportError:
    from ptliq.model.utils import (
        build_graph_inputs_for_split,
        inject_portfolio_dependent_targets as inject_targets,  # type: ignore
    )

# GNN training loop (already in your repo and used by tests)
from ptliq.training.gnn_loop import GNNTrainConfig, train_gnn


app = typer.Typer(no_args_is_help=True, add_completion=False)


# ------------------------
# Experiment configuration
# ------------------------
@dataclass
class ExpConfig:
    # simulation size
    n_bonds: int = 1500
    n_days: int = 30
    providers: List[str] = None
    seed: int = 11

    # split days (YYYY-MM-DD)
    train_end: str = "2025-01-18"
    val_end: str = "2025-01-24"

    # GNN training
    gnn_device: str = "auto"
    gnn_max_epochs: int = 40
    gnn_patience: int = 6
    gnn_batch_size: int = 1024
    gnn_lr: float = 3e-4
    gnn_n_layers: int = 2
    gnn_nhead: int = 4
    gnn_d_model: int = 128
    gnn_hidden: int = 64
    gnn_out_dim: int = 128

    # Baseline training (CPU)
    base_max_epochs: int = 30
    base_patience: int = 5
    base_batch_size: int = 2048
    base_lr: float = 1e-3
    base_hidden: Tuple[int, ...] = (128,)

    def __post_init__(self):
        if self.providers is None:
            self.providers = ["P1", "P2"]


def _now() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _mk_dirs(workdir: Path, run_id: str):
    rawdir = workdir / "data" / "raw" / "sim"
    interim_dir = workdir / "data" / "interim"
    models_dir = workdir / "models"
    reports_dir = workdir / "reports" / run_id
    rawdir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    return rawdir, interim_dir, models_dir, reports_dir


def _simulate(rawdir: Path, ec: ExpConfig):
    frames = simulate(
        SimParams(
            n_bonds=ec.n_bonds,
            n_days=ec.n_days,
            providers=ec.providers,
            seed=ec.seed,
            outdir=rawdir,
        )
    )
    (rawdir / "bonds.parquet").write_bytes(frames["bonds"].to_parquet(index=False))
    (rawdir / "trades.parquet").write_bytes(frames["trades"].to_parquet(index=False))
    print(f"SIM → {rawdir}")


def _split(rawdir: Path, interim_dir: Path, train_end: str, val_end: str) -> Path:
    ranges = compute_default_ranges(rawdir / "trades.parquet", train_end=train_end, val_end=val_end)
    stamp = _now()
    out = interim_dir / "splits" / stamp
    rpath = write_ranges(ranges, out)
    print(f"SPLIT → {rpath}")
    return rpath


# ------------------------
# Baseline: simple MLP on flattened features from GraphInputs
# ------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: Tuple[int, ...] = (128,)):
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _one_hot(x: torch.Tensor, num_classes: int) -> torch.Tensor:
    # x: [B], int64
    oh = torch.zeros(x.shape[0], num_classes, dtype=torch.float32, device=x.device)
    oh.scatter_(1, x.view(-1, 1), 1.0)
    return oh


def _vectorize_baseline(gi, card_sector: int, card_rating: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorize GraphInputs into a per-trade matrix without portfolio context.
    X = [nums, one_hot(sector_code), one_hot(rating_code)]
    y = synthetic y already injected into gi.y
    """
    nums = gi.nums.to(device).to(torch.float32)  # [B, n_num]
    sec = gi.cats["sector_code"].to(device).long()
    rat = gi.cats["rating_code"].to(device).long()
    X = torch.cat(
        [nums, _one_hot(sec, card_sector), _one_hot(rat, card_rating)],
        dim=1,
    )
    y = gi.y.to(device).to(torch.float32).squeeze(-1)
    return X, y


@torch.no_grad()
def _mae(pred: torch.Tensor, true: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(pred - true)).item())


def _train_baseline(
    gi_tr, gi_va, gi_te, ec: ExpConfig
) -> Dict[str, float]:
    # Decide device (baseline on CPU is fine)
    device = torch.device("cpu")

    # Cardinalities for one-hot
    card_sector = int(gi_tr.node_to_sector.max().item() + 1)
    card_rating = int(gi_tr.cats["rating_code"].max().item() + 1)

    Xtr, ytr = _vectorize_baseline(gi_tr, card_sector, card_rating, device)
    Xva, yva = _vectorize_baseline(gi_va, card_sector, card_rating, device)
    Xte, yte = _vectorize_baseline(gi_te, card_sector, card_rating, device)

    in_dim = Xtr.shape[1]
    model = MLP(in_dim, hidden=ec.base_hidden).to(device)
    opt = optim.Adam(model.parameters(), lr=ec.base_lr)
    loss_fn = nn.L1Loss()

    best = {"epoch": 0, "val_mae": 1e9}
    bad = 0
    max_epochs = ec.base_max_epochs
    bs = ec.base_batch_size

    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(Xtr.shape[0])
        for i in range(0, Xtr.shape[0], bs):
            idx = perm[i:i+bs]
            xb, yb = Xtr[idx], ytr[idx]
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        # validate
        model.eval()
        with torch.no_grad():
            pred_val = torch.empty_like(yva)
            for i in range(0, Xva.shape[0], bs):
                sl = slice(i, i+bs)
                pred_val[sl] = model(Xva[sl])
            val_mae = _mae(pred_val, yva)

        if val_mae + 1e-6 < best["val_mae"]:
            best = {"epoch": epoch, "val_mae": val_mae, "state_dict": {k: v.cpu() for k, v in model.state_dict().items()}}
            bad = 0
        else:
            bad += 1
            if bad >= ec.base_patience:
                break

    # load best
    model.load_state_dict(best["state_dict"])
    model.eval()

    # test MAE
    with torch.no_grad():
        pred_te = torch.empty_like(yte)
        for i in range(0, Xte.shape[0], bs):
            sl = slice(i, i+bs)
            pred_te[sl] = model(Xte[sl])
        test_mae = _mae(pred_te, yte)

    return {
        "best_epoch": int(best["epoch"]),
        "best_val_mae": float(best["val_mae"]),
        "test_mae": float(test_mae),
    }


# ------------------------
# GNN: train using your existing loop; evaluate on test
# ------------------------
def _train_and_eval_gnn(rawdir: Path, rpath: Path, models_dir: Path, run_id: str, ec: ExpConfig) -> Dict[str, float]:
    gi_tr = build_graph_inputs_for_split(rawdir, rpath, "train", max_port_items=96)
    gi_va = build_graph_inputs_for_split(rawdir, rpath, "val", max_port_items=96)
    gi_te = build_graph_inputs_for_split(rawdir, rpath, "test", max_port_items=96)

    gi_tr = inject_targets(gi_tr)
    gi_va = inject_targets(gi_va)
    gi_te = inject_targets(gi_te)

    outdir = models_dir / f"{run_id}_gnn"
    outdir.mkdir(parents=True, exist_ok=True)

    gcfg = GNNTrainConfig(
        device=ec.gnn_device,
        max_epochs=ec.gnn_max_epochs,
        patience=ec.gnn_patience,
        batch_size=ec.gnn_batch_size,
        lr=ec.gnn_lr,
        n_layers=ec.gnn_n_layers,
        nhead=ec.gnn_nhead,
        d_model=ec.gnn_d_model,
        gnn_out_dim=ec.gnn_out_dim,
        gnn_num_hidden=ec.gnn_hidden,
        use_baseline=False,
    )
    res = train_gnn(gi_tr, gi_va, outdir, gcfg)
    print(f"GNN TRAIN → {outdir} (best_epoch={res['best_epoch']})")

    # Evaluate on test
    dev = torch.device("cuda" if ec.gnn_device in ["auto", "cuda"] and torch.cuda.is_available() else "cpu")
    ckpt = torch.load(outdir / "ckpt_gnn.pt", map_location="cpu")

    # Recreate model and run forward in batches
    from ptliq.model import PortfolioResidualModel, ModelConfig, NodeFieldSpec
    cat_specs = [
        NodeFieldSpec("sector_code", int(gi_tr.node_to_sector.max().item() + 1), 16),
        NodeFieldSpec("rating_code", int(gi_tr.cats["rating_code"].max().item() + 1), 12),
    ]
    mcfg = ModelConfig(
        n_nodes=gi_tr.n_nodes,
        node_id_dim=32,
        cat_specs=cat_specs,
        n_num=int(gi_tr.nums.shape[1]),
        gnn_num_hidden=ec.gnn_hidden,
        gnn_out_dim=ec.gnn_out_dim,
        d_model=ec.gnn_d_model,
        nhead=ec.gnn_nhead,
        n_layers=ec.gnn_n_layers,
        device=ec.gnn_device,
        use_baseline=False,
    )
    model = PortfolioResidualModel(mcfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(dev).eval()

    with torch.no_grad():
        B = int(gi_te.node_ids.shape[0])
        bs = 1024
        preds, trues = [], []
        for i in range(0, B, bs):
            sl = slice(i, i + bs)
            out = model(
                gi_te.node_ids[sl].to(dev),
                {"sector_code": gi_te.cats["sector_code"][sl].to(dev),
                 "rating_code": gi_te.cats["rating_code"][sl].to(dev)},
                gi_te.nums[sl].to(dev),
                {k: v.to(dev) for k, v in gi_te.issuer_groups.items()},
                {k: v.to(dev) for k, v in gi_te.sector_groups.items()},
                gi_te.node_to_issuer.to(dev),
                gi_te.node_to_sector.to(dev),
                gi_te.port_nodes[sl].to(dev),
                gi_te.port_len[sl].to(dev),
                None,
                None,
            )
            preds.append(out["mean"].detach().cpu())
            trues.append(gi_te.y[sl].cpu())
        pred = torch.cat(preds, dim=0).squeeze(-1)
        true = torch.cat(trues, dim=0).squeeze(-1)
        test_mae = float(torch.mean(torch.abs(pred - true)).item())

    return {
        "best_epoch": int(res["best_epoch"]),
        "test_mae": float(test_mae),
        "model_dir": str(outdir),
    }


# ------------------------
# CLI
# ------------------------
@app.command()
def run(
    config: Path = typer.Option(..., "--config", help="Experiment YAML; supports project.paths + data.sim + split + gnn_train"),
    workdir: Path = typer.Option(Path("."), "--workdir", help="Working directory root"),
):
    """
    Run a realistic experiment that compares:
      - Baseline MLP (no portfolio context) vs
      - GNN+Transformer (uses portfolio composition)

    The target is synthetic and explicitly depends on portfolio composition;
    the expectation is that the GNN outperforms baseline on test MAE.
    """
    # Load config; map fields we need into ExpConfig
    cfg: RootConfig = load_config(config)
    run_id = getattr(cfg.project, "run_id", "exp_gnn_vs_baseline")

    sim = get_sim_config(cfg)
    split = getattr(cfg, "split", {"train_end": "2025-01-18", "val_end": "2025-01-24"})
    gnode = getattr(cfg, "gnn_train", None)

    ec = ExpConfig(
        n_bonds=sim.n_bonds,
        n_days=sim.n_days,
        providers=sim.providers,
        seed=sim.seed,
        train_end=split["train_end"],
        val_end=split["val_end"],
    )
    if gnode is not None:
        ec.gnn_device = gnode.device
        ec.gnn_max_epochs = gnode.max_epochs
        ec.gnn_patience = gnode.patience
        ec.gnn_batch_size = gnode.batch_size
        ec.gnn_lr = gnode.lr
        ec.gnn_n_layers = gnode.n_layers
        ec.gnn_nhead = gnode.nhead
        ec.gnn_d_model = gnode.d_model
        ec.gnn_hidden = gnode.gnn_num_hidden
        ec.gnn_out_dim = gnode.gnn_out_dim

    rawdir, interim_dir, models_dir, reports_dir = _mk_dirs(workdir, run_id)

    # 1) Simulate + split
    _simulate(rawdir, ec)
    rpath = _split(rawdir, interim_dir, ec.train_end, ec.val_end)

    # 2) Build graph inputs for all splits (once) and inject synthetic, portfolio-dependent targets
    gi_tr = build_graph_inputs_for_split(rawdir, rpath, "train", max_port_items=96)
    gi_va = build_graph_inputs_for_split(rawdir, rpath, "val", max_port_items=96)
    gi_te = build_graph_inputs_for_split(rawdir, rpath, "test", max_port_items=96)
    gi_tr = inject_targets(gi_tr)
    gi_va = inject_targets(gi_va)
    gi_te = inject_targets(gi_te)

    # 3) Baseline train/eval
    base_res = _train_baseline(gi_tr, gi_va, gi_te, ec)
    print(f"BASELINE → best_epoch={base_res['best_epoch']}  val_mae={base_res['best_val_mae']:.3f}  test_mae={base_res['test_mae']:.3f}")

    # 4) GNN train/eval (re-use same splits)
    #    Note: train_gnn builds GI again internally; that’s fine and deterministic.
    gnn_res = _train_and_eval_gnn(rawdir, rpath, models_dir, run_id, ec)
    print(f"GNN → best_epoch={gnn_res['best_epoch']}  test_mae={gnn_res['test_mae']:.3f}")

    # 5) Report + relative improvement
    report = {
        "baseline": {
            "best_epoch": int(base_res["best_epoch"]),
            "val_best_mae_bps": float(base_res["best_val_mae"]),
            "test_mae_bps": float(base_res["test_mae"]),
        },
        "gnn": {
            "best_epoch": int(gnn_res["best_epoch"]),
            "test_mae_bps": float(gnn_res["test_mae"]),
            "model_dir": gnn_res["model_dir"],
        },
    }
    report["improvement_vs_baseline_pct"] = float(
        100.0 * (report["baseline"]["test_mae_bps"] - report["gnn"]["test_mae_bps"])
        / max(1e-8, report["baseline"]["test_mae_bps"])
    )

    out = reports_dir / f"gnn_vs_baseline_{_now()}.json"
    out.write_text(json.dumps(report, indent=2))
    print("\n[bold]REPORT[/bold] →", out)
    print(json.dumps(report, indent=2))
