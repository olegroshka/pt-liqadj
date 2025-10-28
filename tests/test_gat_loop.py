from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import pytest

from ptliq.training.gat_loop import (
    SamplerConfig, TrainConfig, ModelConfig, LiquidityRunConfig,
    build_samples, port_collate, train_gat
)


def make_tiny_graph(tmp_path: Path, n: int = 24):
    # nodes
    nodes = pd.DataFrame({
        "node_id": np.arange(n, dtype=int),
        "isin": [f"ISIN{i:05d}" for i in range(n)],
        "rating_num": np.random.randint(1, 20, size=n).astype(float),
        "tenor_years": np.random.uniform(0.5, 15.0, size=n).astype(float),
        "sector_id": np.random.randint(0, 5, size=n).astype(int),
        "curve_bucket": np.random.choice(list("ABC"), size=n),
        "currency": np.random.choice(["USD", "EUR"], size=n),
        "issuer_id": np.random.randint(0, 7, size=n).astype(int),
    })
    # simple chain edges
    src = np.arange(n - 1, dtype=int)
    dst = np.arange(1, n, dtype=int)
    edges = pd.DataFrame({
        "src_id": np.concatenate([src, dst]),
        "dst_id": np.concatenate([dst, src]),
        "relation_id": np.zeros(2 * (n - 1), dtype=int),
        "weight": np.ones(2 * (n - 1), dtype=float),
    })

    base = tmp_path / "features"
    base.mkdir(parents=True, exist_ok=True)
    nodes.to_parquet(base / "graph_nodes.parquet")
    edges.to_parquet(base / "graph_edges.parquet")
    return base, nodes, edges


def make_tiny_trades(tmp_path: Path, nodes: pd.DataFrame, groups: int = 8, lines_per_group: int = 3):
    rng = np.random.default_rng(0)
    isins = nodes["isin"].tolist()
    rows = []
    day0 = np.datetime64("2025-01-01")
    for g in range(groups):
        dt = pd.to_datetime(day0 + np.timedelta64(int(g // 2), 'D'))
        pick = rng.choice(isins, size=lines_per_group + 1, replace=False)
        for i, isin in enumerate(pick):
            sign = 1 if (i % 2 == 0) else -1
            dv01_per_100 = 0.8 + 0.4 * rng.random()
            qty = 1_000_000 + 100_000 * rng.random()
            # simple residual rule to help overfit a bit
            residual = 0.05 * sign
            rows.append({
                "isin": isin,
                "side": "BUY" if sign > 0 else "SELL",
                "dv01_per_100": dv01_per_100,
                "quantity_par": qty,
                "trade_dt": dt,
                "portfolio_id": f"PF_{dt.strftime('%Y%m%d')}_{g:03d}",
                "customer_id": f"C{g%3}",
                "residual": residual,
                "vendor_liq": float(rng.uniform(20, 80)),
            })
    trades = pd.DataFrame(rows)
    p = tmp_path / "trades.parquet"
    trades.to_parquet(p)
    return p


def test_port_collate_shapes():
    # construct two samples with different portfolio lengths
    samples = [
        type("S", (), dict(target_node=1, port_nodes=[2,3], port_weights=[0.2, -0.1], residual=0.1, liq_ref=0.5, dt_ord=1))(),
        type("S", (), dict(target_node=4, port_nodes=[5], port_weights=[0.3], residual=-0.2, liq_ref=0.2, dt_ord=1))(),
    ]
    batch = port_collate(samples)
    assert batch["target_index"].shape == (2,)
    assert batch["port_index"].shape == (3,)
    assert batch["port_batch"].shape == (3,)
    assert batch["port_weight"].shape == (3,)
    assert batch["residual"].shape == (2,)
    assert batch["liq_ref"].shape == (2,)


@pytest.mark.slow
def test_train_gat_integration(tmp_path: Path):
    base, nodes, edges = make_tiny_graph(tmp_path, n=32)
    trades_path = make_tiny_trades(tmp_path, nodes, groups=10, lines_per_group=3)

    # Config: tiny run for CPU
    run_cfg = LiquidityRunConfig(
        sampler=SamplerConfig(expo_scale=5e4, chunk_min=2, chunk_max=4, seed=123),
        train=TrainConfig(device="cpu", max_epochs=5, batch_size=8, lr=1e-3, patience=3, seed=7, print_every=9999, enable_tb=False),
        model=ModelConfig(d_model=64, heads=2, rel_emb_dim=8, issuer_emb_dim=8, dropout=0.1),
    )

    outdir = tmp_path / "model"
    metrics = train_gat(base, trades_path, outdir, run_cfg)

    # Artifacts
    assert (outdir / "ckpt.pt").exists()
    assert (outdir / "ckpt_liquidity.pt").exists()
    assert (outdir / "progress.jsonl").exists()
    assert (outdir / "metrics_val.json").exists()

    # Metrics sanity
    with (outdir / "metrics_val.json").open("r") as f:
        mj = json.load(f)
    assert "best_epoch" in mj and isinstance(mj["best_epoch"], int)

    # Parse progress and ensure finite values and some improvement over epochs if multiple epochs ran
    lines = (outdir / "progress.jsonl").read_text().strip().splitlines()
    assert len(lines) >= 1
    recs = [json.loads(x) for x in lines]
    for r in recs:
        assert np.isfinite(float(r["train_mae"]))
        val = r["val"]
        assert np.isfinite(float(val["mae"]))
        assert float(val["width_mean"]) >= 0.0
    if len(recs) >= 2:
        best = min(r["train_mae"] for r in recs)
        assert best <= recs[0]["train_mae"] + 1e-6
