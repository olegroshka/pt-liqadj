from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Protocol, runtime_checkable, Sequence
import io, json, zipfile
import numpy as np
import torch
import logging

from ptliq.training.loop import load_model_for_eval
import math
import pandas as pd
from typing import Optional

from ptliq.model.mv_dgt import MultiViewDGT
from ptliq.training.mvdgt_loop import MVDGTModelConfig


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
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
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

        # load graph + masks
        view_masks = torch.load(self.workdir / "view_masks.pt", map_location="cpu")
        self.view_masks = {k: v.to(self.device) for k, v in view_masks.items()}

        pyg_graph_path = Path(self.meta["files"]["pyg_graph"])
        data = torch.load(pyg_graph_path, map_location="cpu", weights_only=False)
        self.x = data.x.float().to(self.device)
        self.edge_index = data.edge_index.to(self.device)
        self.edge_weight = data.edge_weight.to(self.device) if hasattr(data, "edge_weight") else None
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
        
        # Optional market preprocessing (z-score + orientation sign)
        self._mkt_mean = None
        self._mkt_std = None
        self._mkt_sign = 1.0
        try:
            preproc_path = self.workdir / "market_preproc.json"
            if preproc_path.exists():
                pre = json.loads(preproc_path.read_text())
                self._mkt_mean = torch.tensor(pre.get("mean", []), dtype=torch.float32, device=self.device)
                std_vals = [float(v) if (v is not None and float(v) > 0.0) else 1.0 for v in pre.get("std", [])]
                self._mkt_std = torch.tensor(std_vals, dtype=torch.float32, device=self.device)
                self._mkt_sign = float(pre.get("sign", 1.0))
        except Exception:
            # fall back silently if unavailable
            self._mkt_mean, self._mkt_std, self._mkt_sign = None, None, 1.0
        
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
        ).to(self.device)
        self.model.load_state_dict(self.state_dict, strict=True)
        self.model.eval()

    @classmethod
    def from_dir(cls, workdir: Path | str, device: Optional[str] = None) -> "DGTScorer":
        return cls(workdir=workdir, device=device)

    @torch.no_grad()
    def score_many(self, rows: List[Dict[str, Any]]) -> np.ndarray:
        if not rows:
            return np.zeros((0,), dtype=np.float32)

        # map isins
        node_ids = []
        for r in rows:
            isin = str(r.get("isin", "")).strip()
            if isin not in self._isin_to_node:
                raise ValueError(f"Unknown ISIN: {isin}")
            node_ids.append(self._isin_to_node[isin])
        anchor_idx = torch.as_tensor(node_ids, dtype=torch.long, device=self.device)

        # trade features (raw → standardized using train-time scaler.json)
        side = [_side_sign(r.get("side")) for r in rows]
        lsz = [_log_size(r.get("size")) for r in rows]
        side_t = torch.as_tensor(side, device=self.device, dtype=torch.float32)
        lsz_t = torch.as_tensor(lsz, device=self.device, dtype=torch.float32)
        n = side_t.size(0)
        feat_tensors = []
        for i, name in enumerate(self.bundle.feature_names):
            if name == "side_sign":
                t = side_t
            elif name == "log_size":
                t = lsz_t
            else:
                # Unknown feature name: impute with scaler mean for that slot
                t = torch.full((n,), float(self._scaler_mean[i].item()), device=self.device, dtype=torch.float32)
            feat_tensors.append(t)
        raw = torch.stack(feat_tensors, dim=1) if feat_tensors else torch.zeros((n, 0), device=self.device)
        denom = torch.where(self._scaler_std <= 0, torch.ones_like(self._scaler_std), self._scaler_std)
        trade = (raw - self._scaler_mean) / denom
        trade = torch.nan_to_num(trade, nan=0.0, posinf=0.0, neginf=0.0)

        # market slice (exact match first, else nearest prior)
        market_feat = None
        if self.mkt_ctx is not None:
            idxs: List[int] = []
            last_idx = int(self.mkt_ctx["mkt_feat"].size(0) - 1)
            dates_index = getattr(self, "_mkt_dates", None)
            idx_arr = getattr(self, "_mkt_idxs", None)
            for r in rows:
                ts = _to_date(r.get("asof_date"))
                if (ts is None) or (self.mkt_lookup is None):
                    idxs.append(last_idx)
                    continue
                if ts in self.mkt_lookup:
                    idxs.append(int(self.mkt_lookup[ts]))
                    continue
                if (dates_index is None) or (idx_arr is None) or (len(idx_arr) == 0):
                    idxs.append(last_idx)
                else:
                    pos = int(dates_index.searchsorted(ts, side="right") - 1)
                    pos = max(0, min(pos, len(idx_arr) - 1))
                    idxs.append(int(idx_arr[pos]))
            di = torch.as_tensor(idxs, dtype=torch.long, device=self.device)
            market_feat = self.mkt_ctx["mkt_feat"].index_select(0, di)

        # --- pf_gid: prefer explicit in request; else derive from training port_ctx; else -1
        pf_list: List[int] = []
        any_explicit = any(("pf_gid" in r) and (r["pf_gid"] is not None) for r in rows)
        if any_explicit:
            for r in rows:
                v = r.get("pf_gid", -1)
                try:
                    pf_list.append(int(v))
                except Exception:
                    pf_list.append(-1)
        else:
            for nid in node_ids:
                pf_list.append(int(self._node_to_pfgid.get(int(nid), -1)))
        pf_gid = torch.as_tensor(pf_list, dtype=torch.long, device=self.device)

        yhat = self.model(
            self.x,
            anchor_idx=anchor_idx,
            market_feat=market_feat,
            pf_gid=pf_gid,
            port_ctx=self.port_ctx,
            trade_feat=trade,
        )
        y = yhat.detach().cpu().numpy().astype(np.float32)
        y = y.reshape(-1)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return y
