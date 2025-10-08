from pathlib import Path
from .config import RootConfig

def ensure_dirs(cfg: RootConfig) -> None:
    for p in [cfg.paths.raw_dir, cfg.paths.interim_dir, cfg.paths.features_dir, cfg.paths.models_dir, cfg.paths.reports_dir]:
        Path(p).mkdir(parents=True, exist_ok=True)
