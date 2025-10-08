from __future__ import annotations
from pydantic import BaseModel, Field, validator
from typing import List, Optional

class Paths(BaseModel):
    raw_dir: str = "data/raw/sim"
    interim_dir: str = "data/interim"
    features_dir: str = "data/features"
    models_dir: str = "models"
    reports_dir: str = "reports"
    serving_dir: str = "serving/packages"

class SimConfig(BaseModel):
    n_bonds: int = 200
    n_days: int = 5
    providers: List[str] = ["P1", "P2"]
    seed: int = 123

class SplitConfig(BaseModel):
    train_end: str
    val_end: str

class TrainConfigNode(BaseModel):
    device: str = "cpu"
    max_epochs: int = 10
    batch_size: int = 1024
    lr: float = 1e-3
    patience: int = 3
    hidden: List[int] = [64, 64]
    dropout: float = 0.0
    seed: int = 42

class ProjectConfig(BaseModel):
    name: str = "pt-liqadj"
    run_id: str = "exp001"

class ExperimentConfig(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    paths: Paths = Field(default_factory=Paths)
    data_sim: SimConfig = Field(default_factory=SimConfig)
    split: SplitConfig
    train: TrainConfigNode = Field(default_factory=TrainConfigNode)

    @validator("split")
    def _check_split(cls, v: SplitConfig):
        # keep strict contract from splitter (train_end < val_end)
        return v
