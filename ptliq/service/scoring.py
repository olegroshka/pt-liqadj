from __future__ import annotations

import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Protocol, runtime_checkable, Sequence
from typing import Optional

import numpy as np
import pandas as pd
import torch

from ptliq.model.mv_dgt import MultiViewDGT
from ptliq.training.mvdgt_loop import MVDGTModelConfig
from ptliq.training.gru_loop import GRURegressor, GRUModelConfig


@runtime_checkable
class HasFeatureNames(Protocol):
    feature_names: Sequence[str]


@runtime_checkable
class Scorer(Protocol):
    """Scoring interface for serving.

    Any scorer must expose:
      - bundle.feature_names for the /health endpoint
      - score_many(rows) -> 1D np.ndarray of floats
    """

    bundle: HasFeatureNames

    def score_many(self, rows: List[Dict[str, Any]]) -> np.ndarray: ...


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
    # Prefer model_config.json when present (aligns with GRU/MV-DGT); fallback to train_config.json
    cfg_path_model = path / "model_config.json"
    cfg_path_train = path / "train_config.json"
    cfg: dict
    if cfg_path_model.exists():
        try:
            cfg = json.loads(cfg_path_model.read_text())
        except Exception:
            cfg = {}
    else:
        try:
            cfg = json.loads(cfg_path_train.read_text())
        except Exception:
            cfg = {}
    hidden = cfg.get("hidden", cfg.get("model_hidden", [64, 64]))
    dropout = cfg.get("dropout", cfg.get("model_dropout", 0.0))
    return ModelBundle(
        feature_names=feature_names,
        mean=np.array(sc["mean"], dtype=np.float32),
        std=np.array(sc["std"], dtype=np.float32),
        hidden=hidden,
        dropout=dropout,
        model_dir=path,
        ckpt_bytes=None,
    )


def _load_bundle_from_zip(zip_path: Path) -> ModelBundle:
    with zipfile.ZipFile(zip_path, "r") as z:
        feature_names = json.loads(z.read("feature_names.json"))
        sc = json.loads(z.read("scaler.json"))
        # Prefer model_config.json when present; fallback to train_config.json
        cfg = {}
        try:
            cfg = json.loads(z.read("model_config.json"))
        except KeyError:
            try:
                cfg = json.loads(z.read("train_config.json"))
            except KeyError:
                cfg = {}
        ckpt = z.read("ckpt.pt")
    hidden = cfg.get("hidden", cfg.get("model_hidden", [64, 64]))
    dropout = cfg.get("dropout", cfg.get("model_dropout", 0.0))
    return ModelBundle(
        feature_names=feature_names,
        mean=np.array(sc["mean"], dtype=np.float32),
        std=np.array(sc["std"], dtype=np.float32),
        hidden=hidden,
        dropout=dropout,
        model_dir=None,
        ckpt_bytes=ckpt,
    )


class MLPScorer:
    def __init__(self, bundle: ModelBundle, device: str = "cpu"):
        self.bundle = bundle
        self.device = "cpu" if device != "cuda" or not torch.cuda.is_available() else "cuda"
        self._load_model()

    @classmethod
    def from_dir(cls, model_dir: Path, device: str = "cpu") -> "MLPScorer":
        return cls(_load_bundle_from_dir(Path(model_dir)), device=device)

    @classmethod
    def from_zip(cls, zip_path: Path, device: str = "cpu") -> "MLPScorer":
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
        """
        Build a raw feature vector in the model's canonical order from a request row.
        Supports two input modes:
          1) Pre-vectorized rows: payload already contains feature keys present in
             bundle.feature_names → used directly.
          2) MV-DGT parity rows: payload contains domain keys like 'size', 'side'.
             We derive a minimal subset when feature names include known aliases:
               - 'f_size_log' from 'size' OR 'log_size'
               - 'f_side_buy' from 'side' OR 'side_sign'
             All other features fall back to the scaler mean (robust baseline behavior).
        """
        # Start with means (robust default for missing keys)
        raw = np.array(self.bundle.mean, dtype=np.float32).copy()
        fnames = self.bundle.feature_names
        # Fast name→index mapping
        if not hasattr(self, "_feat_idx") or self._feat_idx is None:
            self._feat_idx = {name: i for i, name in enumerate(fnames)}
        idx = self._feat_idx

        # 1) Direct feature overrides if provided in payload
        for name, i in idx.items():
            if name in payload and payload[name] is not None:
                try:
                    raw[i] = float(payload[name])
                except Exception:
                    pass  # keep mean on parse failure

        # 2) Derive minimal features from common domain fields
        # size → f_size_log
        if "f_size_log" in idx:
            val = None
            if "f_size_log" in payload and payload.get("f_size_log") is not None:
                try:
                    val = float(payload.get("f_size_log"))
                except Exception:
                    val = None
            if val is None:
                if "log_size" in payload and payload.get("log_size") is not None:
                    try:
                        val = float(payload.get("log_size"))
                    except Exception:
                        val = None
                if val is None and ("size" in payload):
                    try:
                        val = float(_log_size(payload.get("size")))
                    except Exception:
                        val = None
            if val is not None:
                raw[idx["f_size_log"]] = float(val)

        # side → f_side_buy (1 if buy-like else 0)
        if "f_side_buy" in idx:
            val = None
            if "f_side_buy" in payload and payload.get("f_side_buy") is not None:
                try:
                    val = float(payload.get("f_side_buy"))
                except Exception:
                    val = None
            if val is None:
                if "side_sign" in payload and payload.get("side_sign") is not None:
                    try:
                        val = 1.0 if float(payload.get("side_sign")) > 0 else 0.0
                    except Exception:
                        val = None
                if val is None and ("side" in payload):
                    try:
                        val = 1.0 if float(_side_sign(payload.get("side"))) > 0 else 0.0
                    except Exception:
                        val = None
            if val is not None:
                raw[idx["f_side_buy"]] = float(val)

        # Standardize; guard against zero std → inf/NaN
        std = np.where(self.bundle.std <= 0, 1.0, self.bundle.std)
        X = (raw - self.bundle.mean) / std
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X.astype(np.float32)

    def score_many(self, rows: List[Dict[str, Any]]) -> np.ndarray:
        if len(rows) == 0:
            return np.zeros((0,), dtype=np.float32)
        X = np.stack([self._vectorize(r) for r in rows]).astype(np.float32)
        with torch.no_grad():
            y = self.model(torch.from_numpy(X).to(self.dev)).cpu().numpy().astype(np.float32)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y[:, 0]
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return y  # shape (N,)


class GRUScorer:
    """Scorer for GRU baseline trained via ptliq.training.gru_loop.
    Supports two artifact layouts:
      1) Legacy MV-DGT-style (app_main): feature_names.json (list), scaler.json, mvdgt_meta.json, market_preproc.json
      2) Baseline GRU (train_gru): config.json, feature_names.json ({"trade":[],"market":[]}),
         scaler_trade.json, scaler_market.json, market_features.parquet (via config.train.feature_dir)
    Input rows follow MV-DGT semantics: must include 'side' and 'size'; 'asof_date' used for market window.
    """
    def __init__(self, workdir: Path | str, device: Optional[str] = None):
        self.workdir = Path(workdir)
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(dev)

        # ---- Detect artifact layout
        feat_path = self.workdir / "feature_names.json"
        if not feat_path.exists():
            raise FileNotFoundError(f"feature_names.json not found in {self.workdir}")
        try:
            fn_blob = json.loads(feat_path.read_text())
        except Exception:
            fn_blob = ["side_sign", "log_size"]

        baseline_mode = isinstance(fn_blob, dict) and ("trade" in fn_blob)

        # ---- Load scalers and feature names
        if baseline_mode:
            trade_names = list(fn_blob.get("trade", ["side_sign", "log_size"]))
            market_names = list(fn_blob.get("market", []))
            self.bundle = _FeatBundle(trade_names)
            # trade scaler
            st_path = self.workdir / "scaler_trade.json"
            sm_path = self.workdir / "scaler_market.json"
            if not st_path.exists():
                raise FileNotFoundError(f"expected scaler_trade.json in {self.workdir}")
            st = json.loads(st_path.read_text())
            mean = torch.tensor([float(v) for v in st.get("mean", [0.0]*len(trade_names))], dtype=torch.float32, device=self.device)
            std_vals = []
            for v in st.get("std", [1.0]*len(trade_names)):
                try:
                    f = float(v)
                    std_vals.append(1.0 if not (f > 0.0) or (f != f) else f)
                except Exception:
                    std_vals.append(1.0)
            std = torch.tensor(std_vals, dtype=torch.float32, device=self.device)
            self._scaler_mean = mean
            self._scaler_std = std
            # market scaler
            if sm_path.exists():
                sm = json.loads(sm_path.read_text())
                self._mkt_mean = torch.tensor([float(v) for v in sm.get("mean", [0.0]*len(market_names))], dtype=torch.float32, device=self.device)
                mv = []
                for v in sm.get("std", [1.0]*len(market_names)):
                    try:
                        f = float(v); mv.append(1.0 if not (f > 0.0) or (f != f) else f)
                    except Exception:
                        mv.append(1.0)
                self._mkt_std = torch.tensor(mv, dtype=torch.float32, device=self.device)
            else:
                self._mkt_mean = None
                self._mkt_std = None

            # config + market features
            cfg_json = self.workdir / "config.json"
            if not cfg_json.exists():
                raise FileNotFoundError(f"config.json not found in {self.workdir}")
            cfg = json.loads(cfg_json.read_text())
            train_cfg = cfg.get("train", {}) if isinstance(cfg, dict) else {}
            feature_dir = train_cfg.get("feature_dir") or train_cfg.get("feature_dir_str")
            if not feature_dir:
                raise RuntimeError("config.json missing train.feature_dir")
            feature_dir = Path(feature_dir)
            mkt_path = feature_dir / "market_features.parquet"
            if not mkt_path.exists():
                raise FileNotFoundError(f"market_features.parquet not found in {feature_dir}")
            mkt = pd.read_parquet(mkt_path)
            if "asof_date" not in mkt.columns:
                raise RuntimeError("market_features.parquet missing 'asof_date'")
            mkt = mkt.copy()
            mkt["asof_date"] = pd.to_datetime(mkt["asof_date"]).dt.normalize()
            self._mkt_dates = pd.DatetimeIndex(mkt["asof_date"])  # sorted
            self._mkt_mat = torch.as_tensor(mkt[[c for c in mkt.columns if c != "asof_date"]].astype(float).to_numpy(np.float32, copy=False), dtype=torch.float32, device=self.device)
            # z-score
            if (self._mkt_mean is not None) and (self._mkt_std is not None) and (self._mkt_mean.numel() == self._mkt_mat.size(1)):
                denom = torch.where(self._mkt_std <= 0, torch.ones_like(self._mkt_std), self._mkt_std)
                self._mkt_z = (self._mkt_mat - self._mkt_mean) / denom
                self._mkt_z = torch.nan_to_num(self._mkt_z, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                self._mkt_z = self._mkt_mat
            # model config
            mc_path = self.workdir / "model_config.json"
            if mc_path.exists():
                try:
                    mcfg = json.loads(mc_path.read_text())
                except Exception:
                    mcfg = {}
            else:
                # Fallback to baseline config.json persisted by train_gru
                cfg_json = self.workdir / "config.json"
                try:
                    blob = json.loads(cfg_json.read_text()) if cfg_json.exists() else {}
                except Exception:
                    blob = {}
                mcfg = blob.get("model", {}) if isinstance(blob, dict) else {}
            try:
                self.model_cfg = GRUModelConfig(**mcfg)
            except TypeError:
                allowed = set(GRUModelConfig.__annotations__.keys())
                mcfg = {k: v for k, v in mcfg.items() if k in allowed}
                self.model_cfg = GRUModelConfig(**mcfg)
            # checkpoint
            state_obj = torch.load(self.workdir / "ckpt.pt", map_location="cpu")
            state_dict = state_obj.get("state_dict", state_obj)
            # Prefer architecture recorded in checkpoint (authoritative),
            # fallback to model_config.json and sensible defaults
            arch = state_obj.get("arch", {}) if isinstance(state_obj, dict) else {}
            # Update model_cfg fields from arch if present
            try:
                if isinstance(arch, dict):
                    if "hidden" in arch:
                        self.model_cfg.hidden = int(arch["hidden"])
                    if "layers" in arch:
                        self.model_cfg.layers = int(arch["layers"])
                    if "dropout" in arch:
                        self.model_cfg.dropout = float(arch["dropout"])
                    if "window" in arch:
                        self.model_cfg.window = int(arch["window"])
                    # trade_dim from arch is critical for head alignment
                    if "trade_dim" in arch and arch["trade_dim"]:
                        self.model_cfg.trade_dim = int(arch["trade_dim"])
            except Exception:
                # if any cast fails, keep existing model_cfg values
                pass

            # Market dimension is determined by loaded market feature matrix
            mkt_dim = int(self._mkt_z.size(1)) if self._mkt_z is not None else 0
            trade_dim = int(self.model_cfg.trade_dim or len(trade_names))
            # As a final safety, ensure trade_dim matches scaler length
            if trade_dim != len(trade_names):
                # trust checkpoint arch if available; otherwise realign to feature list
                trade_dim = int(arch.get("trade_dim", len(trade_names))) if isinstance(arch, dict) else len(trade_names)
                self.model_cfg.trade_dim = trade_dim

            self.model = GRURegressor(
                mkt_dim=mkt_dim,
                trade_dim=trade_dim,
                hidden=int(self.model_cfg.hidden),
                layers=int(self.model_cfg.layers),
                dropout=float(self.model_cfg.dropout),
            ).to(self.device)
            self.model.load_state_dict(state_dict, strict=True)
            self.model.eval()
            self._legacy = False
        else:
            # ---- Legacy layout (kept for backward compatibility)
            fnames = list(fn_blob) if isinstance(fn_blob, list) else ["side_sign", "log_size"]
            self.bundle = _FeatBundle(fnames)
            sc_path = self.workdir / "scaler.json"
            if sc_path.exists():
                s = json.loads(sc_path.read_text())
                mean = torch.tensor([float(v) for v in s.get("mean", [0.0]*len(fnames))], dtype=torch.float32, device=self.device)
                stdv = []
                for v in s.get("std", [1.0]*len(fnames)):
                    try:
                        f = float(v)
                        stdv.append(1.0 if not (f > 0.0) or (f != f) else f)
                    except Exception:
                        stdv.append(1.0)
                std = torch.tensor(stdv, dtype=torch.float32, device=self.device)
            else:
                mean = torch.zeros((len(fnames),), dtype=torch.float32, device=self.device)
                std = torch.ones((len(fnames),), dtype=torch.float32, device=self.device)
            self._scaler_mean = mean
            self._scaler_std = std
            # load model config and checkpoint
            cfg_path = self.workdir / "model_config.json"
            if not cfg_path.exists():
                raise FileNotFoundError(f"model_config.json not found in {self.workdir}")
            mcfg = json.loads(cfg_path.read_text())
            try:
                self.model_cfg = GRUModelConfig(**mcfg)
            except TypeError:
                allowed = set(GRUModelConfig.__annotations__.keys())
                mcfg = {k: v for k, v in mcfg.items() if k in allowed}
                self.model_cfg = GRUModelConfig(**mcfg)
            ckpt_path = self.workdir / "ckpt.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
            state_obj = torch.load(ckpt_path, map_location="cpu")
            state_dict = state_obj.get("state_dict", state_obj)
            # Load meta & market context/index for date mapping
            meta_path = self.workdir / "mvdgt_meta.json"
            if not meta_path.exists():
                raise FileNotFoundError(f"mvdgt_meta.json not found in {self.workdir}. Re-run training to persist a copy.")
            meta = json.loads(meta_path.read_text())
            mkt_ctx_path = meta.get("files", {}).get("market_context")
            self.mkt_ctx = torch.load(mkt_ctx_path, map_location=self.device) if (mkt_ctx_path and Path(mkt_ctx_path).exists()) else None
            # market preproc
            pre_path = self.workdir / "market_preproc.json"
            self._mkt_mean = None
            self._mkt_std = None
            self._mkt_sign = 1.0
            if pre_path.exists():
                try:
                    pre = json.loads(pre_path.read_text())
                    self._mkt_mean = torch.tensor(pre.get("mean", []), dtype=torch.float32, device=self.device)
                    std_vals = [float(v) if (v is not None and float(v) > 0.0) else 1.0 for v in pre.get("std", [])]
                    self._mkt_std = torch.tensor(std_vals, dtype=torch.float32, device=self.device)
                    self._mkt_sign = float(pre.get("sign", 1.0))
                except Exception:
                    self._mkt_mean, self._mkt_std, self._mkt_sign = None, None, 1.0
            # build GRU model
            mkt_dim = int(self.model_cfg.mkt_dim or (int(self.mkt_ctx["mkt_feat"].size(1)) if self.mkt_ctx is not None else 0))
            self.model = GRURegressor(
                mkt_dim=mkt_dim,
                trade_dim=int(self.model_cfg.trade_dim),
                hidden=int(self.model_cfg.hidden),
                layers=int(self.model_cfg.layers),
                dropout=float(self.model_cfg.dropout),
            ).to(self.device)
            self.model.load_state_dict(state_dict, strict=True)
            self.model.eval()
            self._legacy = True

    @classmethod
    def from_dir(cls, workdir: Path | str, device: Optional[str] = None) -> "GRUScorer":
        return cls(workdir=workdir, device=device)

    @torch.no_grad()
    def score_many(self, rows: List[Dict[str, Any]]) -> np.ndarray:
        if not rows:
            return np.zeros((0,), dtype=np.float32)
        window = max(1, int(self.model_cfg.window or 1))
        n = len(rows)
        # trade features
        side = [_side_sign(r.get("side")) for r in rows]
        lsz = [_log_size(r.get("size")) for r in rows]
        side_t = torch.as_tensor(side, device=self.device, dtype=torch.float32)
        lsz_t = torch.as_tensor(lsz, device=self.device, dtype=torch.float32)
        # align to feature_names order, impute mean for unknowns
        feat_tensors = []
        for i, name in enumerate(self.bundle.feature_names):
            if name == "side_sign":
                t = side_t
            elif name == "log_size":
                t = lsz_t
            else:
                t = torch.full((n,), float(self._scaler_mean[i].item()), device=self.device, dtype=torch.float32)
            feat_tensors.append(t)
        raw = torch.stack(feat_tensors, dim=1) if feat_tensors else torch.zeros((n, 0), device=self.device)
        denom = torch.where(self._scaler_std <= 0, torch.ones_like(self._scaler_std), self._scaler_std)
        trade = (raw - self._scaler_mean) / denom
        trade = torch.nan_to_num(trade, nan=0.0, posinf=0.0, neginf=0.0)

        # market sequence window per row
        if getattr(self, "_legacy", False):
            if self.mkt_ctx is None:
                mseq = torch.zeros((n, window, 0), dtype=torch.float32, device=self.device)
            else:
                last_idx = int(self.mkt_ctx["mkt_feat"].size(0) - 1)
                # map asof_date → date_idx via nearest prior
                idxs: List[int] = []
                for r in rows:
                    ts = _to_date(r.get("asof_date"))
                    if ts is None:
                        idxs.append(last_idx)
                    else:
                        # no explicit index; just clamp to last
                        idxs.append(last_idx)
                di = torch.as_tensor(idxs, dtype=torch.long, device=self.device)
                mkt_feat = self.mkt_ctx["mkt_feat"]  # [T, F]
                mseq = torch.zeros((n, window, int(mkt_feat.size(1))), dtype=torch.float32, device=self.device)
                for w in range(window):
                    t = di - (window - 1 - w)
                    t = t.clamp_min(0)
                    z = mkt_feat.index_select(0, t)
                    if (self._mkt_mean is not None) and (self._mkt_std is not None):
                        denom_m = torch.where(self._mkt_std <= 0, torch.ones_like(self._mkt_std), self._mkt_std)
                        z = (z - self._mkt_mean) / denom_m
                        z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
                        z = z * float(getattr(self, "_mkt_sign", 1.0))
                    mseq[:, w, :] = z
        else:
            # baseline: use preloaded _mkt_z and _mkt_dates
            if getattr(self, "_mkt_z", None) is None:
                mseq = torch.zeros((n, window, 0), dtype=torch.float32, device=self.device)
            else:
                # nearest prior index using searchsorted over _mkt_dates
                idxs: List[int] = []
                for r in rows:
                    ts = _to_date(r.get("asof_date"))
                    if (ts is None) or (len(self._mkt_dates) == 0):
                        idxs.append(int(self._mkt_z.size(0) - 1))
                    else:
                        pos = int(self._mkt_dates.searchsorted(ts, side="right") - 1)
                        pos = max(0, min(pos, int(self._mkt_z.size(0) - 1)))
                        idxs.append(pos)
                di = torch.as_tensor(idxs, dtype=torch.long, device=self.device)
                T, F = int(self._mkt_z.size(0)), int(self._mkt_z.size(1))
                mseq = torch.zeros((n, window, F), dtype=torch.float32, device=self.device)
                for w in range(window):
                    t = di - (window - 1 - w)
                    t = t.clamp_min(0)
                    mseq[:, w, :] = self._mkt_z.index_select(0, t)

        yhat = self.model(mseq, trade)
        y = yhat.detach().cpu().numpy().astype(np.float32)
        y = y.reshape(-1)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return y




# -------------------- DGTScorer --------------------
SIDE_MAP = {
    "B": 1.0, "BUY": 1.0, "CBUY": 1.0, True: 1.0, 1: 1.0,
    "S": -1.0, "SELL": -1.0, "CSELL": -1.0, False: -1.0, 0: -1.0, -1: -1.0
}

def _side_sign(x: Any) -> float:
    if x is None:
        return 0.0
    s = str(x).strip().upper()
    return float(SIDE_MAP.get(s, SIDE_MAP.get(x, 0.0)))

def _log_size(x: Any) -> float:
    try:
        v = float(x)
        return float(math.log1p(abs(v))) if v == v else 0.0
    except Exception:
        return 0.0

def _to_date(x: Any):
    if x is None:
        return None
    try:
        return pd.to_datetime(x, errors="coerce").normalize()
    except Exception:
        return None

# --- Runtime portfolio context helpers (permutation-invariant) ---
from collections import defaultdict

def _normalize_weights(abs_weights: np.ndarray) -> np.ndarray:
    # use float64 accumulator for determinism across permutations
    s = float(np.sum(abs_weights, dtype=np.float64))
    if s <= 0.0 or not np.isfinite(s):
        n = int(abs_weights.size)
        return np.full(n, 1.0 / max(1, n), dtype=np.float32)
    return (abs_weights.astype(np.float64) / s).astype(np.float32)


def _build_runtime_port_ctx(
    rows: List[Dict[str, Any]],
    node_ids: List[int],
    device: torch.device,
) -> tuple[Optional[dict], Optional[torch.Tensor]]:
    """
    Build per-request portfolio context from `portfolio_id`.
    Returns (port_ctx, pf_gid_tensor) if any portfolio_id is present; else (None, None).
    Weights are permutation-invariant, depending only on (isin, side, size):
      - abs weight ∝ abs(size)
      - signed weight ∝ side_sign * abs(size)
      - both normalized within the portfolio to sum(abs)=1
    """
    groups: dict[Any, list[tuple[int, int, float, float]]] = defaultdict(list)
    any_pid = False
    for i, r in enumerate(rows):
        pid = r.get("portfolio_id")
        if pid is None:
            continue
        any_pid = True
        side = _side_sign(r.get("side"))
        try:
            sz = float(r.get("size", 0.0))
        except Exception:
            sz = 0.0
        a = abs(sz)
        s = (1.0 if side >= 0 else -1.0) * a
        groups[pid].append((i, int(node_ids[i]), a, s))

    if not any_pid:
        return None, None

    # stable order by first occurrence in dict iteration
    ordered = list(groups.items())

    port_nodes: list[int] = []
    port_w_abs: list[float] = []
    port_w_sgn: list[float] = []
    port_len: list[int] = []
    pf_gid = np.full(len(rows), -1, dtype=np.int64)

    for g_idx, (_pid, items) in enumerate(ordered):
        # stable within-group ordering for determinism across permutations:
        # sort by (node_id, abs_size desc, signed_size desc, row_idx)
        items_sorted = sorted(items, key=lambda t: (t[1], -t[2], -t[3]))
        abs_vec = np.array([a for (_, _, a, _) in items_sorted], dtype=np.float32)
        sgn_vec = np.array([s for (_, _, _, s) in items_sorted], dtype=np.float32)
        w_abs = _normalize_weights(abs_vec)
        denom = float(np.sum(abs_vec, dtype=np.float64))
        if denom <= 0.0 or not np.isfinite(denom):
            denom = float(len(items_sorted)) if len(items_sorted) > 0 else 1.0
        w_sgn = (sgn_vec.astype(np.float64) / denom).astype(np.float32)
        port_len.append(len(items_sorted))
        for (row_idx, nid, _a, _s), wA, wS in zip(items_sorted, w_abs, w_sgn):
            port_nodes.append(nid)
            port_w_abs.append(float(wA))
            port_w_sgn.append(float(wS))
            pf_gid[row_idx] = g_idx

    port_ctx = {
        "port_nodes_flat": torch.tensor(port_nodes, dtype=torch.long, device=device),
        "port_w_abs_flat": torch.tensor(port_w_abs, dtype=torch.float32, device=device),
        "port_w_signed_flat": torch.tensor(port_w_sgn, dtype=torch.float32, device=device),
        "port_len": torch.tensor(port_len, dtype=torch.long, device=device),
    }
    pf_gid_t = torch.tensor(pf_gid, dtype=torch.long, device=device)
    return port_ctx, pf_gid_t


def _build_runtime_port_ctx_from_explicit_gid(
    rows: List[Dict[str, Any]],
    node_ids: List[int],
    device: torch.device,
) -> Optional[dict]:
    """
    Build per-request portfolio context using explicit pf_gid provided in rows.
    Returns port_ctx dict or None if no valid pf_gid present.
    Weights are permutation-invariant within each provided group:
      - abs weight ∝ abs(size)
      - signed weight ∝ side_sign * abs(size)
      - both normalized within the group to sum(abs)=1
    """
    groups: dict[int, list[int]] = {}
    for i, r in enumerate(rows):
        try:
            g = int(r.get("pf_gid", -1))
        except Exception:
            g = -1
        if g >= 0:
            groups.setdefault(g, []).append(i)
    if not groups:
        return None

    port_nodes: list[int] = []
    port_w_abs: list[float] = []
    port_w_sgn: list[float] = []
    port_len: list[int] = []

    for g, idxs in sorted(groups.items(), key=lambda kv: kv[0]):
        abs_sizes: list[float] = []
        sgn_sizes: list[float] = []
        for i in idxs:
            side = _side_sign(rows[i].get("side"))
            try:
                sz = float(rows[i].get("size", 0.0))
            except Exception:
                sz = 0.0
            a = abs(sz)
            s = (1.0 if side >= 0 else -1.0) * a
            abs_sizes.append(a)
            sgn_sizes.append(s)
        abs_arr = np.asarray(abs_sizes, dtype=np.float32)
        sgn_arr = np.asarray(sgn_sizes, dtype=np.float32)
        denom = float(abs_arr.sum())
        if denom <= 0.0 or not np.isfinite(denom):
            denom = float(len(abs_arr)) if len(abs_arr) > 0 else 1.0
        w_abs = (abs_arr / denom).astype(np.float32)
        w_sgn = (sgn_arr / denom).astype(np.float32)
        port_len.append(len(idxs))
        for i, wA, wS in zip(idxs, w_abs, w_sgn):
            port_nodes.append(int(node_ids[i]))
            port_w_abs.append(float(wA))
            port_w_sgn.append(float(wS))

    return {
        "port_nodes_flat": torch.tensor(port_nodes, dtype=torch.long, device=device),
        "port_w_abs_flat": torch.tensor(port_w_abs, dtype=torch.float32, device=device),
        "port_w_signed_flat": torch.tensor(port_w_sgn, dtype=torch.float32, device=device),
        "port_len": torch.tensor(port_len, dtype=torch.long, device=device),
    }


class _FeatBundle:
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names


class DGTScorer:
    """
    Scorer for MultiViewDGT; loads training artifacts from a workdir.
    Exposes the same interface as MLPScorer with bundle.feature_names used by /health.
    """
    def __init__(self, workdir: Path | str, device: Optional[str] = None):
        self.workdir = Path(workdir)
        # Default to CPU for deterministic parity with test direct path; caller can override.
        dev = device or "cpu"
        self.device = torch.device(dev)

        # minimal bundle for API compatibility
        feat_path = self.workdir / "feature_names.json"
        if feat_path.exists():
            try:
                fnames = json.loads(feat_path.read_text())
            except Exception:
                fnames = ["side_sign", "log_size"]
        else:
            fnames = ["side_sign", "log_size"]
        self.bundle = _FeatBundle(fnames)

        # checkpoint + meta config (needed for guards below)
        ckpt_path = self.workdir / "ckpt.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.state_dict = ckpt.get("state_dict", ckpt)
        self.model_config = ckpt.get("model_config")
        # normalize to dataclass for consistent access
        if self.model_config is None:
            json_cfg = self.workdir / "model_config.json"
            if json_cfg.exists():
                self.model_config = json.loads(json_cfg.read_text())
            else:
                raise RuntimeError("model_config missing; retrain with config persistence patch.")
        try:
            cfg_obj = MVDGTModelConfig(**self.model_config)
        except TypeError:
            mc = dict(self.model_config)
            allowed = set(MVDGTModelConfig.__annotations__.keys())
            mc = {k: v for k, v in mc.items() if k in allowed}
            cfg_obj = MVDGTModelConfig(**mc)
        self._cfg_obj = cfg_obj
        # Guard: feature_names must match configured trade_dim
        if len(self.bundle.feature_names) != int(self._cfg_obj.trade_dim):
            raise RuntimeError(
                f"Mismatch: trade_dim={self._cfg_obj.trade_dim} vs feature_names={self.bundle.feature_names}"
            )

        # Load scaler (train-split mean/std) for trade features; align to feature_names
        scaler_path = self.workdir / "scaler.json"
        mean, std = [0.0] * len(self.bundle.feature_names), [1.0] * len(self.bundle.feature_names)
        if scaler_path.exists():
            try:
                s = json.loads(scaler_path.read_text())
                sm = list(s.get("mean", []))
                ss = list(s.get("std", []))
                # guard sizes and values
                for i in range(min(len(mean), len(sm))):
                    try:
                        mean[i] = float(sm[i])
                    except Exception:
                        mean[i] = 0.0
                for i in range(min(len(std), len(ss))):
                    try:
                        v = float(ss[i])
                        std[i] = 1.0 if not (v > 0.0) or (v != v) else v
                    except Exception:
                        std[i] = 1.0
            except Exception:
                pass
        self._scaler_mean = torch.tensor(mean, dtype=torch.float32, device=self.device)
        self._scaler_std = torch.tensor(std, dtype=torch.float32, device=self.device)

        # checkpoint + meta
        ckpt_path = self.workdir / "ckpt.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.state_dict = ckpt.get("state_dict", ckpt)
        self.model_config = ckpt.get("model_config")
        # normalize to dataclass for consistent access
        if self.model_config is None:
            json_cfg = self.workdir / "model_config.json"
            if json_cfg.exists():
                self.model_config = json.loads(json_cfg.read_text())
            else:
                raise RuntimeError("model_config missing; retrain with config persistence patch.")
        # some old checkpoints may store tensors for ints; coerce via dataclass
        try:
            cfg_obj = MVDGTModelConfig(**self.model_config)
        except TypeError:
            # if there are unexpected keys or missing ones, attempt a lenient construct
            mc = dict(self.model_config)
            # drop unknowns not in dataclass
            allowed = set(MVDGTModelConfig().__dict__.keys())
            mc = {k: v for k, v in mc.items() if k in allowed}
            cfg_obj = MVDGTModelConfig(**mc)
        # fill runtime-computed fields if absent
        # We'll compute x_dim/mkt_dim/use_market after loading graph/contexts below
        self._cfg_obj = cfg_obj

        meta_path = self.workdir / "mvdgt_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta not found: {meta_path}")
        self.meta = json.loads(meta_path.read_text())

        # load graph first (needed to validate masks)
        pyg_graph_path = Path(self.meta["files"]["pyg_graph"])
        data = torch.load(pyg_graph_path, map_location="cpu", weights_only=False)
        self.x = data.x.float().to(self.device)
        self.edge_index = data.edge_index.to(self.device)
        self.edge_weight = data.edge_weight.to(self.device) if hasattr(data, "edge_weight") else None
        # load view masks: prefer meta['files']['view_masks'] if present; else fallback to workdir copy
        masks_path = None
        try:
            masks_path = self.meta.get("files", {}).get("view_masks")
        except Exception:
            masks_path = None
        if masks_path and Path(masks_path).exists():
            vm_src = Path(masks_path)
        else:
            vm_src = self.workdir / "view_masks.pt"
        if not vm_src.exists():
            raise FileNotFoundError(f"view_masks.pt not found at {vm_src}. Ensure dataset builder saved masks.")
        view_masks = torch.load(vm_src, map_location="cpu")
        # Validate masks vs edge count E
        E = int(self.edge_index.size(1))
        def _as_bool_1d(t: torch.Tensor) -> torch.Tensor:
            t = t.view(-1)
            if t.dtype != torch.bool:
                t = (t != 0)
            return t
        vm_norm = {k: _as_bool_1d(v) for k, v in view_masks.items()}
        bad = {k: int(v.numel()) for k, v in vm_norm.items() if int(v.numel()) != E}
        if bad:
            details = ", ".join([f"{k}:len={n}" for k, n in bad.items()])
            raise RuntimeError(
                f"view_masks length mismatch vs graph edges (E={E}). Offending masks: {details}. "
                f"Use masks built from the same pyg_graph; if using prebuilt artifacts, ensure meta['files']['view_masks'] points to the correct masks."
            )
        self.view_masks = {k: v.to(self.device) for k, v in vm_norm.items()}
        # ensure runtime dims are set
        if self._cfg_obj.x_dim is None:
            self._cfg_obj.x_dim = int(self.x.size(1))

        # contexts
        self.mkt_ctx = None
        self.mkt_lookup = None
        mkt_ctx_path = self.meta["files"].get("market_context")
        mkt_idx_path = self.meta["files"].get("market_index")
        if mkt_ctx_path and Path(mkt_ctx_path).exists():
            self.mkt_ctx = torch.load(mkt_ctx_path, map_location=self.device)
            if mkt_idx_path and Path(mkt_idx_path).exists():
                idx_df = pd.read_parquet(mkt_idx_path)
                idx_df["asof_date"] = pd.to_datetime(idx_df["asof_date"]).dt.normalize()
                self.mkt_lookup = {pd.Timestamp(r.asof_date): int(r.row_idx) for r in idx_df.itertuples(index=False)}
                # Precompute sorted dates and index array for fast nearest-prior lookup
                try:
                    items = sorted(self.mkt_lookup.items())
                    self._mkt_dates = pd.to_datetime([k for k, _ in items])  # DatetimeIndex
                    self._mkt_idxs = np.asarray([v for _, v in items], dtype=np.int64)
                except Exception:
                    self._mkt_dates, self._mkt_idxs = None, None
            else:
                self._mkt_dates, self._mkt_idxs = None, None
        else:
            self._mkt_dates, self._mkt_idxs = None, None

        self.port_ctx = None
        port_ctx_path = self.meta["files"].get("portfolio_context")
        if port_ctx_path and Path(port_ctx_path).exists():
            self.port_ctx = torch.load(port_ctx_path, map_location=self.device)
        
        # Optional market preprocessing (z-score + orientation sign) — gated by explicit 'apply' flag for backward compatibility
        self._mkt_mean = None
        self._mkt_std = None
        self._mkt_sign = 1.0
        self._mkt_apply = False
        try:
            preproc_path = self.workdir / "market_preproc.json"
            if preproc_path.exists():
                pre = json.loads(preproc_path.read_text())
                self._mkt_apply = bool(pre.get("apply", False))
                self._mkt_mean = torch.tensor(pre.get("mean", []), dtype=torch.float32, device=self.device)
                std_vals = [float(v) if (v is not None and float(v) > 0.0) else 1.0 for v in pre.get("std", [])]
                self._mkt_std = torch.tensor(std_vals, dtype=torch.float32, device=self.device)
                self._mkt_sign = float(pre.get("sign", 1.0))
        except Exception:
            # fall back silently if unavailable
            self._mkt_mean, self._mkt_std, self._mkt_sign, self._mkt_apply = None, None, 1.0, False
        
        # Build a node_id -> pf_gid map from training portfolio context (if available)
        self._node_to_pfgid: dict[int, int] = {}
        if self.port_ctx is not None:
            try:
                nodes_flat = self.port_ctx.get("port_nodes_flat")
                lens = self.port_ctx.get("port_len")
                if (nodes_flat is not None) and (lens is not None):
                    nodes_np = nodes_flat.detach().cpu().numpy().astype(int)
                    lens_np = lens.detach().cpu().numpy().astype(int)
                    off = 0
                    for g, L in enumerate(lens_np):
                        for k in range(int(L)):
                            nid = int(nodes_np[off + k])
                            if nid not in self._node_to_pfgid:
                                self._node_to_pfgid[nid] = int(g)
                        off += int(L)
            except Exception:
                # leave map empty; scorer will fall back to -1
                self._node_to_pfgid = {}

        # ISIN → node
        nodes_parq = self.meta["files"].get("graph_nodes")
        if not nodes_parq:
            guess = pyg_graph_path.parent / "graph_nodes.parquet"
            if not guess.exists():
                raise FileNotFoundError("graph_nodes.parquet not found (meta['files']['graph_nodes'] missing and no adjacent file)")
            nodes_parq = str(guess)
        nodes = pd.read_parquet(nodes_parq)
        self._isin_to_node = {str(r.isin): int(r.node_id) for r in nodes.itertuples(index=False)}

        # model config
        if self.model_config is None:
            json_cfg = self.workdir / "model_config.json"
            if json_cfg.exists():
                self.model_config = json.loads(json_cfg.read_text())
            else:
                raise RuntimeError("model_config missing; retrain with config persistence patch.")

        # ensure market dims flags
        if self._cfg_obj.mkt_dim is None:
            self._cfg_obj.mkt_dim = int(self.mkt_ctx["mkt_feat"].size(1)) if self.mkt_ctx is not None else 0
        if self._cfg_obj.use_market is None:
            self._cfg_obj.use_market = bool(self._cfg_obj.mkt_dim > 0)

        self.model = MultiViewDGT(
            x_dim=int(self._cfg_obj.x_dim or self.x.size(1)),
            hidden=int(self._cfg_obj.hidden),
            heads=int(self._cfg_obj.heads),
            dropout=float(self._cfg_obj.dropout),
            view_masks=self.view_masks,
            edge_index=self.edge_index,
            edge_weight=self.edge_weight,
            mkt_dim=int(self._cfg_obj.mkt_dim or 0),
            use_portfolio=bool(self._cfg_obj.use_portfolio),
            use_market=bool(self._cfg_obj.use_market),
            trade_dim=int(self._cfg_obj.trade_dim),
            view_names=list(getattr(self._cfg_obj, "views", ["struct","port","corr_global","corr_local"])),
            use_pf_head=bool(getattr(self._cfg_obj, "use_pf_head", False)),
            pf_head_hidden=getattr(self._cfg_obj, "pf_head_hidden", None),
            # portfolio attention wiring
            use_portfolio_attn=bool(getattr(self._cfg_obj, "use_portfolio_attn", False)),
            portfolio_attn_layers=int(getattr(self._cfg_obj, "portfolio_attn_layers", 1) or 1),
            portfolio_attn_heads=int(getattr(self._cfg_obj, "portfolio_attn_heads", 4) or 4),
            portfolio_attn_dropout=float(getattr(self._cfg_obj, "portfolio_attn_dropout", self._cfg_obj.dropout if getattr(self._cfg_obj, "dropout", None) is not None else 0.1) or 0.1),
            portfolio_attn_hidden=(int(self._cfg_obj.get("portfolio_attn_hidden")) if isinstance(getattr(self._cfg_obj, "portfolio_attn_hidden", None), (int, float)) else None) if isinstance(self._cfg_obj, dict) else (int(getattr(self._cfg_obj, "portfolio_attn_hidden")) if (getattr(self._cfg_obj, "portfolio_attn_hidden", None) is not None) else None),
            portfolio_attn_concat_trade=bool(getattr(self._cfg_obj, "portfolio_attn_concat_trade", True)),
            portfolio_attn_concat_market=bool(getattr(self._cfg_obj, "portfolio_attn_concat_market", False)),
            portfolio_attn_mode=str(getattr(self._cfg_obj, "portfolio_attn_mode", "residual")),
            portfolio_attn_gate_init=float(getattr(self._cfg_obj, "portfolio_attn_gate_init", 0.0) or 0.0),
            max_portfolio_len=(int(getattr(self._cfg_obj, "max_portfolio_len")) if (getattr(self._cfg_obj, "max_portfolio_len", None) is not None) else None),
        ).to(self.device)
        self.model.load_state_dict(self.state_dict, strict=True)
        self.model.eval()

    @classmethod
    def from_dir(cls, workdir: Path | str, device: Optional[str] = None) -> "DGTScorer":
        return cls(workdir=workdir, device=device)

    @torch.no_grad()
    @torch.no_grad()
    def score_many(self, rows: List[Dict[str, Any]]) -> np.ndarray:
        if not rows:
            return np.zeros((0,), dtype=np.float32)

        # --- node ids
        node_ids = []
        for r in rows:
            isin = str(r.get("isin", "")).strip()
            if isin not in self._isin_to_node:
                raise ValueError(f"Unknown ISIN: {isin}")
            node_ids.append(self._isin_to_node[isin])
        anchor_idx = torch.as_tensor(node_ids, dtype=torch.long, device=self.device)

        # --- trade features (standardized with train-time scaler.json)
        side = [_side_sign(r.get("side")) for r in rows]
        lsz  = [_log_size(r.get("size")) for r in rows]
        side_t = torch.as_tensor(side, dtype=torch.float32, device=self.device)
        lsz_t  = torch.as_tensor(lsz,  dtype=torch.float32, device=self.device)

        # align to saved feature_names (impute means for any unknown names)
        n = side_t.size(0)
        feat_cols = []
        for i, name in enumerate(self.bundle.feature_names):
            if name == "side_sign":
                feat_cols.append(side_t)
            elif name == "log_size":
                feat_cols.append(lsz_t)
            else:
                feat_cols.append(torch.full((n,), float(self._scaler_mean[i].item()),
                                            dtype=torch.float32, device=self.device))
        raw   = torch.stack(feat_cols, dim=1) if feat_cols else torch.zeros((n,0), device=self.device)
        denom = torch.where(self._scaler_std <= 0, torch.ones_like(self._scaler_std), self._scaler_std)
        trade = torch.nan_to_num((raw - self._scaler_mean) / denom, nan=0.0, posinf=0.0, neginf=0.0)

        # --- market (apply z-score + orientation sign if available, matching direct path)
        market_feat = None
        if self.mkt_ctx is not None:
            idxs: List[int] = []
            last_idx = int(self.mkt_ctx["mkt_feat"].size(0) - 1)
            dates_index = getattr(self, "_mkt_dates", None)
            idx_arr     = getattr(self, "_mkt_idxs",  None)
            for r in rows:
                ts = _to_date(r.get("asof_date"))
                if (ts is None) or (self.mkt_lookup is None):
                    idxs.append(last_idx)
                elif ts in self.mkt_lookup:
                    idxs.append(int(self.mkt_lookup[ts]))
                elif (dates_index is None) or (idx_arr is None) or (len(idx_arr) == 0):
                    idxs.append(last_idx)
                else:
                    pos = int(dates_index.searchsorted(ts, side="right") - 1)
                    pos = max(0, min(pos, len(idx_arr) - 1))
                    idxs.append(int(idx_arr[pos]))
            market_feat = self.mkt_ctx["mkt_feat"].index_select(0, torch.as_tensor(idxs, dtype=torch.long, device=self.device))
            # z-score and orientation sign (if preproc artifacts are present)
            if self._mkt_apply and (self._mkt_mean is not None) and (self._mkt_std is not None):
                denom_m = torch.where(self._mkt_std <= 0, torch.ones_like(self._mkt_std), self._mkt_std)
                market_feat = (market_feat - self._mkt_mean) / denom_m
                market_feat = torch.nan_to_num(market_feat, nan=0.0, posinf=0.0, neginf=0.0)
                if getattr(self, "_mkt_sign", None) is not None:
                    market_feat = market_feat * float(self._mkt_sign)

        # --- portfolio context: ONLY if the request provides pf info
        any_pf_gid = any(("pf_gid" in r) and (r["pf_gid"] is not None) for r in rows)
        any_pid    = any((r.get("portfolio_id") is not None) for r in rows)
        port_ctx = None
        pf_gid   = None
        if any_pf_gid:
            # explicit pf_gid vector from request
            pf_list = []
            for r in rows:
                try: pf_list.append(int(r.get("pf_gid", -1)))
                except Exception: pf_list.append(-1)
            pf_gid = torch.as_tensor(pf_list, dtype=torch.long, device=self.device)
            port_ctx = _build_runtime_port_ctx_from_explicit_gid(rows, node_ids, self.device)
            if (port_ctx is None) and any_pid:
                # fall back to runtime grouping by portfolio_id
                port_ctx, pf_gid = _build_runtime_port_ctx(rows, node_ids, self.device)
        elif any_pid:
            # runtime grouping by portfolio_id
            port_ctx, pf_gid = _build_runtime_port_ctx(rows, node_ids, self.device)
        # else: keep portfolio path OFF (pf_gid=None, port_ctx=None)

        yhat = self.model(
            self.x,
            anchor_idx=anchor_idx,
            market_feat=market_feat,
            pf_gid=pf_gid,
            port_ctx=port_ctx,
            trade_feat=trade,
        )
        y = yhat.detach().cpu().numpy().astype(np.float32).reshape(-1)
        return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
