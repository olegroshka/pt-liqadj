# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import numpy as np
import torch
from typer.testing import CliRunner

from ptliq.cli.gnn_build_dataset import app as gnn_build_dataset_app
from ptliq.cli.gnn_build_graph import app as gnn_build_graph_app
from ptliq.data.simulate import simulate, SimParams
from ptliq.data.split import compute_default_ranges, write_ranges
from ptliq.model.utils import GraphInputs
from ptliq.training.gnn_loop import (
    train_gnn,
    GNNTrainConfig,
    PortfolioResidualModelLike,
)

runner = CliRunner()


# ---------- helpers ----------
def _safe_torch_load(p: Path):
    from torch.serialization import add_safe_globals, safe_globals
    add_safe_globals([GraphInputs])
    with safe_globals([GraphInputs]):
        return torch.load(p, map_location="cpu")


def _mk_tiny_splits(trades_path: Path, outdir: Path) -> Path:
    ranges = compute_default_ranges(trades_path, train_end="2025-01-03", val_end="2025-01-04")
    return write_ranges(ranges, outdir)


def _ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)


def _infer_schema_example(workdir: Path) -> GraphInputs:
    """
    Build a tiny real bundle and return it so tests can adapt to the live schema.
    Ensures 'workdir' exists to avoid OSError when pandas writes parquet.
    """
    _ensure_dir(workdir)

    frames = simulate(SimParams(n_bonds=40, n_days=3, providers=["P1"], seed=9, outdir=workdir))
    bonds_pq = workdir / "bonds.parquet"
    trades_pq = workdir / "trades.parquet"
    frames["bonds"].to_parquet(bonds_pq, index=False)
    frames["trades"].to_parquet(trades_pq, index=False)

    splits_file = write_ranges(
        compute_default_ranges(trades_pq, train_end="2025-01-02", val_end="2025-01-03"),
        workdir / "splits",
    )
    splits_dir = splits_file.parent

    graph_dir = workdir / "graph"
    r = runner.invoke(gnn_build_graph_app, ["--bonds", str(bonds_pq), "--outdir", str(graph_dir)])
    assert r.exit_code == 0, r.stdout or r.stderr

    ds_dir = workdir / "gnn_ds"
    r = runner.invoke(
        gnn_build_dataset_app,
        [
            "--trades", str(trades_pq),
            "--bonds", str(bonds_pq),
            "--splits-dir", str(splits_dir),
            "--graph-dir", str(graph_dir),
            "--outdir", str(ds_dir),
            "--derive-target",
            "--ref-price-col", "clean_price",
            "--default-par", "100",
        ],
    )
    assert r.exit_code == 0, r.stdout or r.stderr
    gi = _safe_torch_load(ds_dir / "train.pt")
    assert isinstance(gi, GraphInputs)
    return gi


def _nums_first_tensor(nums: Any) -> Optional[torch.Tensor]:
    """Return a representative numeric tensor from nums (tensor or dict-of-tensors)."""
    if nums is None:
        return None
    if torch.is_tensor(nums):
        return nums
    if isinstance(nums, dict):
        for v in nums.values():
            if torch.is_tensor(v):
                return v
        return None
    return None


# ================================================================
# 1) DATASET SHAPE/CONSISTENCY TEST (bundle integrity)
# ================================================================
def test_gnn_dataset_bundle_integrity(tmp_path: Path):
    # --- simulate tiny dataset ---
    frames = simulate(SimParams(n_bonds=100, n_days=4, providers=["P1"], seed=123, outdir=tmp_path))
    bonds_pq = tmp_path / "bonds.parquet"
    trades_pq = tmp_path / "trades.parquet"
    frames["bonds"].to_parquet(bonds_pq, index=False)
    frames["trades"].to_parquet(trades_pq, index=False)

    # --- splits ---
    splits_file = _mk_tiny_splits(trades_pq, tmp_path / "splits")
    splits_dir = splits_file.parent

    # --- build graph (CLI) ---
    graph_dir = tmp_path / "graph"
    res = runner.invoke(gnn_build_graph_app, ["--bonds", str(bonds_pq), "--outdir", str(graph_dir)])
    assert res.exit_code == 0, res.stdout or res.stderr

    # --- build dataset bundles (CLI) ---
    ds_dir = tmp_path / "gnn_ds"
    res = runner.invoke(
        gnn_build_dataset_app,
        [
            "--trades", str(trades_pq),
            "--bonds", str(bonds_pq),
            "--splits-dir", str(splits_dir),
            "--graph-dir", str(graph_dir),
            "--outdir", str(ds_dir),
            "--derive-target",
            "--ref-price-col", "clean_price",
            "--default-par", "100",
        ],
    )
    assert res.exit_code == 0, res.stdout or res.stderr

    # --- load bundles & check shapes/fields ---
    tr = _safe_torch_load(ds_dir / "train.pt")
    va = _safe_torch_load(ds_dir / "val.pt")
    te = _safe_torch_load(ds_dir / "test.pt")

    for gi in (tr, va, te):
        assert isinstance(gi, GraphInputs)
        n = int(gi.node_ids.shape[0])
        assert n > 0

        # cats: dict[str -> Tensor] with matching first dim
        assert isinstance(gi.cats, dict)
        for k, v in gi.cats.items():
            assert isinstance(v, torch.Tensor), k
            assert v.shape[0] == n, (k, v.shape, n)

        # nums: either None, a Tensor [n, d], or a dict[str->Tensor] with all tensors having [n, *]
        if gi.nums is not None:
            if torch.is_tensor(gi.nums):
                assert gi.nums.shape[0] == n
            elif isinstance(gi.nums, dict):
                got_any = False
                for k, v in gi.nums.items():
                    if torch.is_tensor(v):
                        got_any = True
                        assert v.shape[0] == n, (k, v.shape, n)
                assert got_any, "nums dict had no tensors"
            else:
                raise AssertionError(f"Unexpected nums type: {type(gi.nums)}")

        # y: [n]
        assert gi.y.shape[0] == n

        # portfolio fields (tolerate absent/empty; if present, validate shape)
        if getattr(gi, "port_nodes", None) is not None and torch.is_tensor(gi.port_nodes) and gi.port_nodes.numel() > 0:
            assert gi.port_nodes.shape[0] == n
        if getattr(gi, "port_len", None) is not None and torch.is_tensor(gi.port_len) and gi.port_len.numel() > 0:
            assert gi.port_len.shape[0] == n

        # mappings/relations must exist
        assert gi.node_to_issuer is not None
        assert gi.node_to_sector is not None
        assert isinstance(gi.issuer_groups, dict) and len(gi.issuer_groups) > 0
        assert isinstance(gi.sector_groups, dict) and len(gi.sector_groups) > 0

        # optional fields
        if hasattr(gi, "size_side_urg") and getattr(gi, "size_side_urg") is not None:
            ssu = getattr(gi, "size_side_urg")
            assert torch.is_tensor(ssu) and ssu.shape[0] == n


# ================================================================
# 2) TRAINING-LOOP BEHAVIOR on SYNTHETIC GRAPH (improves MAE)
# ================================================================
class TinyLinearModel(torch.nn.Module):
    """
    Minimal model matching PortfolioResidualModelLike contract:
    - uses numeric features (first block if dict) to predict y
    - returns dict with 'mean'
    - defines compute_loss
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.lin = torch.nn.Linear(in_dim, 1)

    def forward(  # type: ignore[override]
        self,
        batch_nodes: torch.Tensor,
        batch_cats: Dict[str, torch.Tensor],
        batch_nums: Optional[Any],
        issuer_groups: Optional[Dict[int, torch.Tensor]],
        sector_groups: Optional[Dict[int, torch.Tensor]],
        node_to_issuer: Optional[torch.Tensor],
        node_to_sector: Optional[torch.Tensor],
        port_nodes: Optional[torch.Tensor],
        port_len: Optional[torch.Tensor],
        *args,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        x = _nums_first_tensor(batch_nums)
        assert x is not None, "This toy model needs numeric features"
        pred = self.lin(x)  # [B, 1]
        return {"mean": pred}

    def compute_loss(self, out: Dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(out["mean"].squeeze(-1), target)


def _mk_synth_graphinputs(schema_ref: GraphInputs, n: int = 512, d: int = 6, seed: int = 7) -> Tuple[GraphInputs, GraphInputs]:
    """
    Make train/val GraphInputs where the true target is a linear function of nums.
    Adapt to the live schema (nums tensor vs dict; optional fields).
    """
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    w = torch.linspace(-0.8, 0.9, steps=d).unsqueeze(1)  # [d,1]
    y = (X @ w).squeeze(1) + 0.05 * torch.randn(n, generator=g)

    n_tr = int(n * 0.8)
    Xtr, Xva = X[:n_tr], X[n_tr:]
    ytr, yva = y[:n_tr], y[n_tr:]

    # mimic cats presence from schema_ref
    cats_keys = list(schema_ref.cats.keys()) or ["sector_code", "rating_code"]
    cats = {k: torch.zeros(n, dtype=torch.long) for k in cats_keys}
    cats_tr = {k: v[:n_tr] for k, v in cats.items()}
    cats_va = {k: v[n_tr:] for k, v in cats.items()}

    # trivial portfolio info (may be unused): empty to mirror current pipeline tolerance
    port_nodes = torch.empty(0, dtype=torch.long)
    port_len = torch.empty(0, dtype=torch.long)

    # mapping/membership (single issuer/sector group)
    node_to_issuer = torch.zeros(n, dtype=torch.long)
    node_to_sector = torch.zeros(n, dtype=torch.long)
    issuer_groups = {0: torch.arange(n, dtype=torch.long)}
    sector_groups = {0: torch.arange(n, dtype=torch.long)}

    # respect whether nums is dict or tensor in the live schema
    ref_nums = schema_ref.nums
    def pack_nums(X_):
        if ref_nums is None:
            return None
        if torch.is_tensor(ref_nums):
            return X_.clone()
        if isinstance(ref_nums, dict):
            key = next(iter(ref_nums.keys())) if len(ref_nums) > 0 else "main"
            return {key: X_.clone()}
        return X_.clone()

    def pack(X_, y_, cats_):
        kwargs = dict(
            n_nodes=int(X_.shape[0]),
            node_ids=torch.arange(X_.shape[0], dtype=torch.long),
            cats=cats_,
            nums=pack_nums(X_),
            issuer_groups=issuer_groups,
            sector_groups=sector_groups,
            node_to_issuer=node_to_issuer[: X_.shape[0]],
            node_to_sector=node_to_sector[: X_.shape[0]],
            port_nodes=port_nodes,  # may be empty
            port_len=port_len,      # may be empty
            y=y_.clone(),
        )
        if hasattr(schema_ref, "size_side_urg"):
            kwargs["size_side_urg"] = None
        return GraphInputs(**kwargs)

    gi_tr = pack(Xtr, ytr, cats_tr)
    gi_va = pack(Xva, yva, cats_va)
    return gi_tr, gi_va


def test_gnn_loop_learns_on_synthetic(tmp_path: Path):
    # infer live schema from a tiny real bundle
    schema_ref = _infer_schema_example(tmp_path / "schema_probe")

    # synth train/val where y is linear in nums
    train_gi, val_gi = _mk_synth_graphinputs(schema_ref, n=480, d=5, seed=99)
    x_val = _nums_first_tensor(val_gi.nums)
    assert x_val is not None

    cfg = GNNTrainConfig(
        device="cpu",
        max_epochs=40,
        batch_size=64,
        lr=5e-3,
        patience=10,
        seed=123,
    )

    # factory to build our tiny linear model
    in_dim = int(_nums_first_tensor(train_gi.nums).shape[1])
    def factory(_gi: GraphInputs, _cfg: GNNTrainConfig) -> PortfolioResidualModelLike:
        return TinyLinearModel(in_dim=in_dim)

    # baseline MAE before training
    with torch.no_grad():
        init_model = factory(train_gi, cfg)
        pred0 = init_model(
            val_gi.node_ids, val_gi.cats, val_gi.nums,
            val_gi.issuer_groups, val_gi.sector_groups,
            val_gi.node_to_issuer, val_gi.node_to_sector,
            getattr(val_gi, "port_nodes", None), getattr(val_gi, "port_len", None),
        )["mean"].squeeze(-1).detach().cpu().numpy()
        yva = val_gi.y.detach().cpu().numpy()
        init_mae = float(np.mean(np.abs(pred0 - yva)))

    metrics = train_gnn(train_gi, val_gi, tmp_path / "models/gnn_test", cfg, model_factory=factory)

    # artifacts
    mdir = tmp_path / "models/gnn_test"
    assert (mdir / "ckpt_gnn.pt").exists()
    assert (mdir / "ckpt.pt").exists()
    assert (mdir / "metrics_val.json").exists()

    best = json.loads((mdir / "metrics_val.json").read_text())
    assert best["best_epoch"] >= 1
    assert np.isfinite(best["best_val_mae_bps"])

    # learned something meaningful
    assert best["best_val_mae_bps"] < init_mae * 0.6
