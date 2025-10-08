# ptliq/utils/config.py  (append/modify)

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import yaml
from pydantic import BaseModel, Field, ValidationError

class SimConfig(BaseModel):
    n_bonds: int = Field(200, ge=10)
    n_days: int = Field(10, ge=1)
    providers: list[str] = Field(default_factory=lambda: ["P1", "P2"])
    seed: int = 42

class Paths(BaseModel):
    raw_dir: Path = Path("data/raw")
    interim_dir: Path = Path("data/interim")
    features_dir: Path = Path("data/features")
    models_dir: Path = Path("models")
    reports_dir: Path = Path("reports")

class Project(BaseModel):
    name: str = "pt-liqadj"
    seed: int = 42
    run_id: str = "exp001"  # NEW: default run id

# NEW: split + train nodes
class SplitConfig(BaseModel):
    train_end: str
    val_end: str

class TrainConfigNode(BaseModel):
    device: str = "cpu"
    max_epochs: int = 10
    batch_size: int = 1024
    lr: float = 1e-3
    patience: int = 3
    hidden: list[int] = Field(default_factory=lambda: [64, 64])
    dropout: float = 0.0
    seed: int = 42

class RootConfig(BaseModel):
    project: Project = Project()
    paths: Paths = Paths()
    data: Dict[str, Any] = Field(default_factory=dict)
    # keep top-level split/train flexible (don’t break older configs)
    split: Dict[str, Any] = Field(default_factory=dict)
    train: Dict[str, Any] = Field(default_factory=dict)

def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def load_config(path: str | Path) -> RootConfig:
    raw = load_yaml(path)
    try:
        return RootConfig(**raw)
    except ValidationError as e:
        raise SystemExit(f"[config] invalid config {path}:\n{e}") from e

def get_sim_config(cfg: RootConfig) -> SimConfig:
    sim = cfg.data.get("sim", {})
    try:
        return SimConfig(**sim)
    except ValidationError as e:
        raise SystemExit(f"[config] invalid data.sim section:\n{e}") from e

# NEW: accessors for split/train
def get_split_config(cfg: RootConfig) -> SplitConfig:
    try:
        return SplitConfig(**cfg.split)
    except ValidationError as e:
        raise SystemExit(f"[config] invalid split section:\n{e}") from e

def get_train_config(cfg: RootConfig) -> TrainConfigNode:
    try:
        return TrainConfigNode(**cfg.train)
    except ValidationError as e:
        raise SystemExit(f"[config] invalid train section:\n{e}") from e
