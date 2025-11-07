from __future__ import annotations

import json
from pathlib import Path
import os
import typer
import torch
import yaml

from ptliq.training.mvdgt_loop import MVDGTTrainConfig, MVDGTModelConfig, train_mvdgt as _train

app = typer.Typer(no_args_is_help=True)


def _merge_dicts(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for k, v in (overlay or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


def _load_default_yaml() -> dict | None:
    # prefer CWD path; fallback to project-relative path if running from repo
    candidates = [
        Path("configs/mvdgt.default.yaml"),
        Path(__file__).resolve().parents[2] / "configs/mvdgt.default.yaml",
    ]
    for p in candidates:
        if p.exists():
            try:
                return yaml.safe_load(p.read_text()) or {}
            except Exception:
                return None
    return None


@app.callback(invoke_without_command=True)
def app_main(
    workdir: Path = typer.Option(Path("data/mvdgt/exp001"), help="Working directory with mvdgt_meta.json, samples, masks"),
    pyg_dir: Path = typer.Option(Path("data/pyg"), help="PyG artifacts dir (pyg_graph.pt, market_index.parquet)"),
    outdir: Path | None = typer.Option(None, help="Model output dir (defaults to workdir when not provided)"),
    epochs: int = typer.Option(5),
    lr: float = typer.Option(1e-3),
    weight_decay: float = typer.Option(1e-4),
    batch_size: int = typer.Option(512),
    seed: int = typer.Option(17),
    device_str: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu"),
    config: Path | None = typer.Option(None, help="Optional YAML config. If omitted, will try configs/mvdgt.default.yaml"),
    no_default_config: bool = typer.Option(False, help="Do not auto-load configs/mvdgt.default.yaml when --config is not provided."),
):
    """Train MV-DGT using the reusable training loop."""
    # When invoked directly as a function in tests, Typer may pass OptionInfo objects
    try:
        from typer.models import OptionInfo as _OptionInfo  # type: ignore
    except Exception:
        class _OptionInfo:  # type: ignore
            pass
    def _norm(v, default):
        return getattr(v, "default", default) if isinstance(v, _OptionInfo) else v

    # normalize possibly-OptionInfo defaults
    epochs = int(_norm(epochs, 5))
    lr = float(_norm(lr, 1e-3))
    weight_decay = float(_norm(weight_decay, 1e-4))
    batch_size = int(_norm(batch_size, 512))
    seed = int(_norm(seed, 17))
    device_str = str(_norm(device_str, "cuda" if torch.cuda.is_available() else "cpu"))
    outdir_opt = _norm(outdir, None)
    # Backward-compatible default: if outdir not provided, save under workdir
    outdir = Path(outdir_opt) if outdir_opt is not None else Path(workdir)

    # normalize possibly-OptionInfo for new options
    config_opt = _norm(config, None)
    no_default_config = bool(_norm(no_default_config, False))

    # --- Load YAML (explicit or default) and merge into config ----
    yaml_cfg: dict | None = None
    if (config_opt is not None) and Path(config_opt).exists():
        try:
            yaml_cfg = yaml.safe_load(Path(config_opt).read_text()) or {}
        except Exception:
            yaml_cfg = None
    elif not no_default_config:
        yaml_cfg = _load_default_yaml()

    # Extract train/model from YAML supporting two shapes:
    # 1) pipeline: data.mvdgt.train / data.mvdgt.model
    # 2) flat: train / model at the root
    yaml_train = {}
    yaml_model = {}
    if isinstance(yaml_cfg, dict):
        if "data" in yaml_cfg and isinstance(yaml_cfg.get("data"), dict):
            mvdgt = (yaml_cfg.get("data") or {}).get("mvdgt") or {}
            yaml_train = (mvdgt.get("train") or {})
            yaml_model = (mvdgt.get("model") or {})
        else:
            yaml_train = yaml_cfg.get("train") or {}
            yaml_model = yaml_cfg.get("model") or {}

    # Build dataclass configs from YAML first
    model_cfg = MVDGTModelConfig(**_merge_dicts(MVDGTModelConfig().__dict__, yaml_model))
    cfg = MVDGTTrainConfig(**_merge_dicts(MVDGTTrainConfig(workdir=workdir, pyg_dir=pyg_dir, outdir=outdir). __dict__, yaml_train))
    # Apply CLI overrides over YAML/dataclass
    cfg.workdir = workdir
    cfg.pyg_dir = pyg_dir
    cfg.outdir = outdir
    cfg.epochs = epochs
    cfg.lr = lr
    cfg.weight_decay = weight_decay
    cfg.batch_size = batch_size
    cfg.seed = seed
    cfg.device = device_str
    # Always set TB dir under outdir unless overridden by YAML
    if not cfg.tb_log_dir:
        cfg.tb_log_dir = str(outdir / "tb")

    # Attach model config
    cfg.model = model_cfg

    metrics = _train(cfg)
    typer.echo(json.dumps(metrics))


if __name__ == "__main__":
    app()
