from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
import io, json, zipfile
import numpy as np
import torch

from ptliq.training.loop import load_model_for_eval

@dataclass
class ModelBundle:
    feature_names: List[str]
    mean: np.ndarray
    std: np.ndarray
    hidden: List[int]
    dropout: float
    model_dir: Path | None  # if loaded from folder (training artifacts)
    ckpt_bytes: bytes | None  # if loaded from zip

def _load_bundle_from_dir(path: Path) -> ModelBundle:
    path = Path(path)
    feature_names = json.loads((path / "feature_names.json").read_text())
    sc = json.loads((path / "scaler.json").read_text())
    cfg = json.loads((path / "train_config.json").read_text())
    return ModelBundle(
        feature_names=feature_names,
        mean=np.array(sc["mean"], dtype=np.float32),
        std=np.array(sc["std"], dtype=np.float32),
        hidden=cfg.get("hidden", [64, 64]),
        dropout=cfg.get("dropout", 0.0),
        model_dir=path,
        ckpt_bytes=None,
    )

def _load_bundle_from_zip(zip_path: Path) -> ModelBundle:
    with zipfile.ZipFile(zip_path, "r") as z:
        feature_names = json.loads(z.read("feature_names.json"))
        sc = json.loads(z.read("scaler.json"))
        cfg = json.loads(z.read("train_config.json"))
        ckpt = z.read("ckpt.pt")
    return ModelBundle(
        feature_names=feature_names,
        mean=np.array(sc["mean"], dtype=np.float32),
        std=np.array(sc["std"], dtype=np.float32),
        hidden=cfg.get("hidden", [64, 64]),
        dropout=cfg.get("dropout", 0.0),
        model_dir=None,
        ckpt_bytes=ckpt,
    )

class Scorer:
    def __init__(self, bundle: ModelBundle, device: str = "cpu"):
        self.bundle = bundle
        self.device = "cpu" if device != "cuda" or not torch.cuda.is_available() else "cuda"
        self._load_model()

    @classmethod
    def from_dir(cls, model_dir: Path, device: str = "cpu") -> "Scorer":
        return cls(_load_bundle_from_dir(Path(model_dir)), device=device)

    @classmethod
    def from_zip(cls, zip_path: Path, device: str = "cpu") -> "Scorer":
        return cls(_load_bundle_from_zip(Path(zip_path)), device=device)

    def _load_model(self) -> None:
        """
        Initialize the MLPRegressor from either a training folder or a packaged zip.
        Robust to:
          - wrapped checkpoints: {"state_dict": ..., "arch": ..., ...}
          - raw state_dict checkpoints
          - missing/empty/bad checkpoints (falls back to random init)
        """
        import io
        import torch

        dev = torch.device(self.device if (self.device == "cuda" and torch.cuda.is_available()) else "cpu")

        feature_names = self.bundle.feature_names
        in_dim = len(feature_names)
        hidden = list(self.bundle.hidden or [])
        dropout = float(self.bundle.dropout or 0.0)

        from ptliq.model.baseline import MLPRegressor
        model = MLPRegressor(in_dim=in_dim, hidden=hidden, dropout=dropout).to(dev)

        def _try_load_state_dict(state_obj) -> bool:
            try:
                state = state_obj
                if isinstance(state_obj, dict) and "state_dict" in state_obj:
                    state = state_obj["state_dict"]
                model.load_state_dict(state, strict=True)
                return True
            except (RuntimeError, KeyError, ValueError):
                return False

        loaded = False

        if self.bundle.model_dir is not None:
            ckpt_path = Path(self.bundle.model_dir) / "ckpt.pt"
            if ckpt_path.exists():
                try:
                    state_obj = torch.load(ckpt_path, map_location=dev)
                    loaded = _try_load_state_dict(state_obj)
                except (EOFError, RuntimeError):
                    loaded = False
        else:
            if self.bundle.ckpt_bytes:
                try:
                    buf = io.BytesIO(self.bundle.ckpt_bytes)
                    state_obj = torch.load(buf, map_location=dev)
                    loaded = _try_load_state_dict(state_obj)
                except (EOFError, RuntimeError):
                    loaded = False

        model.eval()
        self.model = model
        self.dev = dev

    def _vectorize(self, payload: Dict[str, Any]) -> np.ndarray:
        # Raw vector in canonical order; fall back to scaler mean for missing keys.
        raw = np.array(
            [float(payload.get(name, float(self.bundle.mean[i]))) for i, name in enumerate(self.bundle.feature_names)],
            dtype=np.float32,
        )
        # Standardize; guard against zero std → inf/NaN
        X = (raw - self.bundle.mean) / self.bundle.std
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    def score_many(self, rows: List[Dict[str, Any]]) -> np.ndarray:
        if len(rows) == 0:
            return np.zeros((0,), dtype=np.float32)
        X = np.stack([self._vectorize(r) for r in rows]).astype(np.float32)
        with torch.no_grad():
            y = self.model(torch.from_numpy(X).to(self.dev)).cpu().numpy().astype(np.float32)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y[:, 0]
        # ---- sanitize for API/JSON safety ----
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return y  # shape (N,)
