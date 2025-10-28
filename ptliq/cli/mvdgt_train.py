from __future__ import annotations

import json
from pathlib import Path
import os
import typer
import torch

from ptliq.training.mvdgt_loop import MVDGTTrainConfig, train_mvdgt as _train

app = typer.Typer(no_args_is_help=True)


@app.callback(invoke_without_command=True)
def app_main(
    workdir: Path = typer.Option(Path("data/mvdgt/exp001"), help="Working directory with mvdgt_meta.json, samples, masks"),
    pyg_dir: Path = typer.Option(Path("data/pyg"), help="PyG artifacts dir (pyg_graph.pt, market_index.parquet)"),
    outdir: Path = typer.Option(Path(os.getenv("PTLIQ_DEFAULT_MODELS_DIR", "models/mvdgt")), help="Model output dir"),
    epochs: int = typer.Option(5),
    lr: float = typer.Option(1e-3),
    weight_decay: float = typer.Option(1e-4),
    batch_size: int = typer.Option(512),
    seed: int = typer.Option(17),
    device_str: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu"),
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
    outdir = Path(_norm(outdir, Path(os.getenv("PTLIQ_DEFAULT_MODELS_DIR", "models/mvdgt"))))

    cfg = MVDGTTrainConfig(
        workdir=workdir,
        pyg_dir=pyg_dir,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        seed=seed,
        device=device_str,
        tb_log_dir=str(outdir / "tb"),
    )
    metrics = _train(cfg)
    typer.echo(json.dumps(metrics))


if __name__ == "__main__":
    app()
