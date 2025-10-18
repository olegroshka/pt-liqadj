# ptliq/cli/gnn_train.py
from __future__ import annotations

import json
import logging
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import typer
import yaml

from ptliq.training.gnn_loop import GNNTrainConfig, train_gnn
from ptliq.model.utils import GraphInputs  # allowlist for safe unpickling

app = typer.Typer(no_args_is_help=True)

# -------------------------
# Logging
# -------------------------
def _setup_logging(level: str = "INFO") -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)8s | %(name)s | %(message)s",
        datefmt="%-H:%M:%S" if sys.platform != "win32" else "%H:%M:%S",
        stream=sys.stdout,
    )

LOG = logging.getLogger("gnn-train")

# -------------------------
# Safe helpers
# -------------------------
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

def _as_float_maybe(x: Any, default: float) -> float:
    if x is None:
        return float(default)
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x)
        except Exception:
            return float(default)
    return float(default)

def _merge_cfg(js: Dict[str, Any], cli_overrides: Dict[str, Any]) -> GNNTrainConfig:
    js_train = (js.get("train") or {}).copy()
    js_model = (js.get("model") or {}).copy()
    cfg = GNNTrainConfig(
        device=js_train.get("device", "auto"),
        max_epochs=int(js_train.get("max_epochs", 20)),
        batch_size=int(js_train.get("batch_size", 256)),
        lr=_as_float_maybe(js_train.get("lr", 1e-3), 1e-3),
        weight_decay=float(js_train.get("weight_decay", 0.0)),  # <— honor YAML
        patience=int(js_train.get("patience", 3)),
        seed=int(js_train.get("seed", 42)),
        node_id_dim=int(js_model.get("node_id_dim", 32)),
        nhead=int(js_model.get("nhead", 4)),
        n_layers=int(js_model.get("n_layers", 1)),
        d_model=int(js_model.get("d_model", 128)),
        gnn_num_hidden=int(js_model.get("gnn_num_hidden", 64)),
        gnn_out_dim=int(js_model.get("gnn_out_dim", 128)),
        gnn_dropout=float(js_model.get("gnn_dropout", 0.0)),
        head_hidden=int(js_model.get("head_hidden", 128)),
        head_dropout=float(js_model.get("head_dropout", 0.0)),
        use_calibrator=bool(js_model.get("use_calibrator", False)),
        use_baseline=bool(js_model.get("use_baseline", False)),
    )
    for k, v in cli_overrides.items():
        if v is not None and hasattr(cfg, k):
            if k == "lr":
                setattr(cfg, k, _as_float_maybe(v, cfg.lr))
            else:
                setattr(cfg, k, v)
    return cfg

def _safe_load_bundle(p: Path) -> GraphInputs:
    torch.serialization.add_safe_globals([GraphInputs])
    return torch.load(Path(p), map_location="cpu", weights_only=False)

def _nums_dim(gi: GraphInputs) -> int:
    n = gi.nums
    if torch.is_tensor(n):
        return int(n.shape[1]) if n.dim() >= 2 else 1
    if isinstance(n, dict):
        for v in n.values():
            if torch.is_tensor(v):
                return int(v.shape[1]) if v.dim() >= 2 else 1
    return 0

def _preview_dict_tensors(d: Dict[str, torch.Tensor], kmax: int = 3) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for i, (k, v) in enumerate(d.items()):
        if i >= kmax:
            break
        if torch.is_tensor(v):
            out[k] = {"shape": tuple(v.shape), "dtype": str(v.dtype), "min": float(v.min().item()), "max": float(v.max().item())}
        else:
            out[k] = str(type(v))
    return out

# -------------------------
# CLI
# -------------------------
@app.callback(invoke_without_command=True)
def main(
    data_dir: Path = typer.Option(..., help="Folder with train.pt/val.pt/test.pt (from gnn-build-dataset)"),
    outdir: Path = typer.Option(..., help="Where to save checkpoints and metrics"),
    config: Optional[Path] = typer.Option(None, help="YAML config (train/model sections)"),
    # Optional overrides
    device: Optional[str] = typer.Option(None),
    max_epochs: Optional[int] = typer.Option(None),
    batch_size: Optional[int] = typer.Option(None),
    lr: Optional[float] = typer.Option(None),
    patience: Optional[int] = typer.Option(None),
    seed: Optional[int] = typer.Option(None),
    node_id_dim: Optional[int] = typer.Option(None),
    nhead: Optional[int] = typer.Option(None),
    n_layers: Optional[int] = typer.Option(None),
    d_model: Optional[int] = typer.Option(None),
    gnn_num_hidden: Optional[int] = typer.Option(None),
    gnn_out_dim: Optional[int] = typer.Option(None),
    gnn_dropout: Optional[float] = typer.Option(None),
    head_hidden: Optional[int] = typer.Option(None),
    head_dropout: Optional[float] = typer.Option(None),
    use_calibrator: Optional[bool] = typer.Option(None),
    use_baseline: Optional[bool] = typer.Option(None),
    # logging
    loglevel: str = typer.Option("INFO", help="DEBUG|INFO|WARNING|ERROR"),
):
    """
    Train the GNN model (or baseline if use_baseline=True).

    Writes:
      - {outdir}/ckpt_gnn.pt
      - {outdir}/ckpt.pt
      - {outdir}/metrics_val.json
      - {outdir}/progress.jsonl
    """
    _setup_logging(loglevel)
    try:
        data_dir = Path(data_dir)
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        # config
        raw_cfg = _load_yaml(config)
        overrides = {
            "device": device,
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "patience": patience,
            "seed": seed,
            "node_id_dim": node_id_dim,
            "nhead": nhead,
            "n_layers": n_layers,
            "d_model": d_model,
            "gnn_num_hidden": gnn_num_hidden,
            "gnn_out_dim": gnn_out_dim,
            "gnn_dropout": gnn_dropout,
            "head_hidden": head_hidden,
            "head_dropout": head_dropout,
            "use_calibrator": use_calibrator,
            "use_baseline": use_baseline,
        }
        cfg = _merge_cfg(raw_cfg, overrides)

        # echo config & raw yaml
        print("=== gnn-train: effective config ===")
        print(json.dumps(asdict(cfg), indent=2))
        if raw_cfg:
            print("=== gnn-train: raw YAML (train/model) ===")
            print(json.dumps({"train": raw_cfg.get("train"), "model": raw_cfg.get("model")}, indent=2))

        # load bundles
        tr_path = data_dir / "train.pt"
        va_path = data_dir / "val.pt"
        if not tr_path.exists() or not va_path.exists():
            raise FileNotFoundError(f"Missing train/val bundles in {data_dir}")
        train_gi: GraphInputs = _safe_load_bundle(tr_path)
        val_gi: GraphInputs = _safe_load_bundle(va_path)

        # quick data stats
        print("=== gnn-train: dataset stats ===")
        print(json.dumps({
            "n_train": int(train_gi.node_ids.shape[0]),
            "n_val": int(val_gi.node_ids.shape[0]),
            "n_nodes": int(train_gi.n_nodes),
            "n_num": _nums_dim(train_gi),
            "cats": list(train_gi.cats.keys()),
        }, indent=2))

        # preview cats & nums for first few features
        try:
            cats_preview = _preview_dict_tensors(train_gi.cats, kmax=4)
        except Exception:
            cats_preview = {"error": "failed to preview cats"}
        try:
            if torch.is_tensor(train_gi.nums):
                nums_preview = {"tensor": {"shape": tuple(train_gi.nums.shape), "dtype": str(train_gi.nums.dtype)}}
            elif isinstance(train_gi.nums, dict):
                nums_preview = _preview_dict_tensors(train_gi.nums, kmax=4)
            else:
                nums_preview = {"type": str(type(train_gi.nums))}
        except Exception:
            nums_preview = {"error": "failed to preview nums"}
        print("=== gnn-train: preview ===")
        print(json.dumps({"cats": cats_preview, "nums": nums_preview}, indent=2))

        # train (the loop itself logs to progress.jsonl)
        metrics = train_gnn(train_gi, val_gi, outdir, cfg)

        # mirror ckpt & write metrics
        ckpt_main = outdir / "ckpt_gnn.pt"
        ckpt_alias = outdir / "ckpt.pt"
        if ckpt_main.exists():
            try:
                ckpt_alias.write_bytes(ckpt_main.read_bytes())
            except Exception as e:
                print(f"[gnn-train] WARN: failed to mirror ckpt to ckpt.pt: {e}")
        (outdir / "metrics_val.json").write_text(json.dumps(metrics, indent=2))

        print(f"Saved ckpt at {ckpt_main} and {ckpt_alias}; best={metrics}")
    except Exception as e:
        # Print full traceback to STDOUT so Typer's TestRunner captures it
        tb = traceback.format_exc()
        print(f"[gnn-train] ERROR: {e}\n--- traceback ---\n{tb}")
        raise typer.Exit(code=1)
