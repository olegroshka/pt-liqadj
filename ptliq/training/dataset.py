from __future__ import annotations
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

NUM_PREFIX = "f_"

def load_split_parquet(base: Path, split: str) -> pd.DataFrame:
    p = Path(base) / f"{split}.parquet"
    return pd.read_parquet(p)

def discover_features(df: pd.DataFrame) -> List[str]:
    # by convention, features begin with f_
    return sorted([c for c in df.columns if c.startswith(NUM_PREFIX)])

def compute_standardizer(train_df: pd.DataFrame, feat_cols: List[str]) -> Dict[str, np.ndarray]:
    X = train_df[feat_cols].astype(float).to_numpy()
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)  # avoid division by zero
    return {"mean": mu, "std": sd}

def apply_standardizer(df: pd.DataFrame, feat_cols: List[str], stdz: Dict[str, np.ndarray]) -> np.ndarray:
    X = df[feat_cols].astype(float).to_numpy()
    X = (X - stdz["mean"]) / stdz["std"]
    # replace NaNs if any
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X.astype(np.float32)

class FeaturesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.shape[0] == y.shape[0]
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
