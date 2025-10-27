# ptliq/cli/gat_train.py
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional
import os

import typer
import yaml

from ptliq.training.gat_loop import (
    SamplerConfig, TrainConfig, ModelConfig, LiquidityRunConfig, train_gat
)

app = typer.Typer(no_args_is_help=True)


def _load_yaml(p: Optional[Path]) -> Dict[str, Any]:
    if not p:
        return {}
    p = Path(p)
    if not p.exists():
        return {}
    try:
        data = yaml.safe_load(p.read_text()) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _merge_cfg(js: Dict[str, Any], overrides: Dict[str, Any]) -> LiquidityRunConfig:
    js_sampler = (js.get("sampler") or {}).copy()
    js_train = (js.get("train") or {}).copy()
    js_model = (js.get("model") or {}).copy()

    sc = SamplerConfig(**{k: js_sampler.get(k, getattr(SamplerConfig, k)) for k in SamplerConfig.__annotations__})
    tc = TrainConfig(**{k: js_train.get(k, getattr(TrainConfig, k)) for k in TrainConfig.__annotations__})
    mc = ModelConfig(**{k: js_model.get(k, getattr(ModelConfig, k)) for k in ModelConfig.__annotations__})

    # CLI overrides (only when provided)
    for k, v in overrides.items():
        if v is None:
            continue
        if hasattr(tc, k):
            setattr(tc, k, v)
        elif hasattr(mc, k):
            setattr(mc, k, v)
    return LiquidityRunConfig(sampler=sc, train=tc, model=mc)


@app.command()
def app_main(
    features_run_dir: Path = typer.Option(Path(os.getenv("PTLIQ_DEFAULT_PYG_DIR", "data/pyg")), help="Folder with pyg_graph.pt (from featurize pyg)"),
    graph_dir: Path = typer.Option(Path(os.getenv("PTLIQ_DEFAULT_GRAPH_DIR", "data/graph")), help="Folder with graph_nodes/edges artifacts (from featurize graph)"),
    trades: Path = typer.Option(Path(os.getenv("PTLIQ_DEFAULT_TRADES_PATH", "data/raw/sim/trades.parquet"))),
    outdir: Path = typer.Option(Path(os.getenv("PTLIQ_DEFAULT_MODELS_DIR", "models/gat_diff")), help="Model output dir"),
    config: Optional[Path] = typer.Option("configs/gat.default.yaml", help="YAML with {sampler,train,model}"),
    ranges: Optional[Path] = typer.Option(None, help="Optional ranges.json to define train/val periods"),
    # common train overrides
    device: str = typer.Option(None),
    max_epochs: int = typer.Option(None),
    batch_size: int = typer.Option(None),
    lr: float = typer.Option(None),
    patience: int = typer.Option(None),
    seed: int = typer.Option(None),
    # logging options
    tb: Optional[bool] = typer.Option(None, help="Enable TensorBoard logging (default: True)"),
    tb_log_dir: Optional[Path] = typer.Option(None, help="Override TensorBoard log dir (default: <outdir>/tb)"),
):
    """
    Train LiquidityModelGAT (Rel-GATv2 + PMA + cross-attn) from the new graph pipeline.
    """
    raw = _load_yaml(config)
    overrides = dict(device=device, max_epochs=max_epochs, batch_size=batch_size, lr=lr, patience=patience, seed=seed)
    # map logging options into TrainConfig names
    if tb is not None:
        overrides["enable_tb"] = bool(tb)
    if tb_log_dir is not None:
        overrides["tb_log_dir"] = str(tb_log_dir)
    run_cfg = _merge_cfg(raw, overrides)

    print("=== gat-train: effective config ===")
    print(json.dumps({"sampler": asdict(run_cfg.sampler), "train": asdict(run_cfg.train), "model": asdict(run_cfg.model)}, indent=2))

    metrics = train_gat(features_run_dir, trades, outdir, run_cfg, ranges_json=ranges, graph_dir=graph_dir)
    print(f"[OK] saved under {outdir}  |  best={metrics}")
