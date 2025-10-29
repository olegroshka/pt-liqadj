from __future__ import annotations
from pathlib import Path
from typing import Optional
import logging

try:  # pragma: no cover - tensorboard is optional
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:  # pragma: no cover - tensorboard is optional
    SummaryWriter = None  # type: ignore


def get_logger(name: str, workdir: Path, filename: str = "train.log") -> logging.Logger:
    """
    Create (or fetch) a logger configured with file and console handlers.
    - Log file placed under <workdir>/out/<filename> (fallback to <workdir> if out/ cannot be created).
    - Format: "%(asctime)s | %(levelname)s | %(message)s" with YYYY-MM-DD HH:MM:SS.
    - Idempotent: if handlers already exist, does not add duplicates.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    out_dir = Path(workdir) / "out"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        out_dir = Path(workdir)

    log_path = out_dir / filename
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    try:
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:
        # If file handler fails, continue with console only
        pass

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def setup_tb(enable_tb: bool, workdir: Path, tb_log_dir: Optional[str] = None) -> Optional["SummaryWriter"]:
    """
    Initialize a TensorBoard SummaryWriter if enabled and available.
    - When `tb_log_dir` is None, defaults to <workdir>/tb.
    - Returns None gracefully if TB not available or directory cannot be created.
    """
    if not enable_tb or SummaryWriter is None:
        return None
    tb_dir = Path(tb_log_dir) if tb_log_dir else (Path(workdir) / "tb")
    try:
        tb_dir.mkdir(parents=True, exist_ok=True)
        return SummaryWriter(log_dir=str(tb_dir))  # type: ignore[call-arg]
    except Exception:
        return None


def safe_close_tb(writer: Optional["SummaryWriter"]) -> None:
    """Flush and close TB writer safely (no exceptions)."""
    if writer is None:
        return
    try:
        writer.flush()
    except Exception:
        pass
    try:
        writer.close()
    except Exception:
        pass
