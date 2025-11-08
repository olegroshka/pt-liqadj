from __future__ import annotations
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import torch

from ptliq.training.gru_loop import train_gru, GRUTrainConfig, GRUModelConfig
from ptliq.cli.train_baseline import app_main as mlp_train_main
from ptliq.service.scoring import GRUScorer, MLPScorer


def _make_gru_workdir(tmp: Path, T: int = 12, F: int = 4) -> Path:
    work = tmp / "work_gru"
    work.mkdir(parents=True, exist_ok=True)
    # market context tensor [T, F]
    g = torch.Generator().manual_seed(123)
    mkt_feat = torch.randn(T, F, generator=g)
    mkt_path = work / "market_ctx.pt"
    torch.save({"mkt_feat": mkt_feat}, mkt_path)

    # minimal meta with files.market_context (market_index is optional)
    meta = {"files": {"market_context": str(mkt_path)}}
    (work / "mvdgt_meta.json").write_text(json.dumps(meta))

    # samples.parquet with splits
    rng = np.random.default_rng(123)
    n_train, n_val = 256, 64
    date_idx_tr = rng.integers(0, T, size=n_train)
    date_idx_va = rng.integers(0, T, size=n_val)
    side_tr = rng.choice([-1.0, 1.0], size=n_train)
    side_va = rng.choice([-1.0, 1.0], size=n_val)
    logsz_tr = rng.normal(0.0, 1.0, size=n_train)
    logsz_va = rng.normal(0.0, 1.0, size=n_val)

    # planted target in bps scale
    w = torch.randn(F, generator=g) * 0.2
    def _y(di, s, l):
        return float(10.0 * (mkt_feat[int(di)] @ w).item() + 3.0 * float(s) + 1.0 * float(l) + rng.normal(0, 0.1))
    y_tr = [ _y(di, s, l) for di, s, l in zip(date_idx_tr, side_tr, logsz_tr) ]
    y_va = [ _y(di, s, l) for di, s, l in zip(date_idx_va, side_va, logsz_va) ]

    df_tr = pd.DataFrame({
        "split": ["train"] * n_train,
        "date_idx": date_idx_tr.astype(int),
        "side_sign": side_tr.astype(float),
        "log_size": logsz_tr.astype(float),
        "y": y_tr,
    })
    df_va = pd.DataFrame({
        "split": ["val"] * n_val,
        "date_idx": date_idx_va.astype(int),
        "side_sign": side_va.astype(float),
        "log_size": logsz_va.astype(float),
        "y": y_va,
    })
    df = pd.concat([df_tr, df_va], ignore_index=True)
    df.to_parquet(work / "samples.parquet", index=False)
    return work


def _make_mlp_features(tmp: Path, run_id: str = "exp_parity") -> tuple[Path, Path]:
    base = tmp / "features" / run_id
    base.mkdir(parents=True, exist_ok=True)
    rs = np.random.default_rng(0)
    n_train, n_val = 128, 64
    f1_tr = rs.normal(0.0, 1.0, size=n_train)
    f2_tr = rs.normal(0.0, 1.0, size=n_train)
    y_tr = 2.0 * f1_tr - 3.0 * f2_tr + rs.normal(0.0, 0.1, size=n_train)

    f1_va = rs.normal(0.0, 1.0, size=n_val)
    f2_va = rs.normal(0.0, 1.0, size=n_val)
    y_va = 2.0 * f1_va - 3.0 * f2_va + rs.normal(0.0, 0.1, size=n_val)

    df_tr = pd.DataFrame({
        "f_one": f1_tr.astype(float),
        "f_two": f2_tr.astype(float),
        "y_bps": y_tr.astype(float),
    })
    df_va = pd.DataFrame({
        "f_one": f1_va.astype(float),
        "f_two": f2_va.astype(float),
        "y_bps": y_va.astype(float),
    })
    (base / "train.parquet").write_bytes(df_tr.to_parquet(index=False))
    (base / "val.parquet").write_bytes(df_va.to_parquet(index=False))
    return base, base


def _side_sign(x):
    s = str(x).strip().upper()
    if s in ("B", "BUY", "CBUY", "1", "TRUE"): return 1.0
    if s in ("S", "SELL", "CSELL", "0", "FALSE", "-1"): return -1.0
    try:
        v = float(x)
        return 1.0 if v > 0 else (-1.0 if v < 0 else 0.0)
    except Exception:
        return 0.0


def _log_size(x):
    try:
        v = float(x)
        return float(math.log1p(abs(v))) if v == v else 0.0
    except Exception:
        return 0.0


def _make_baseline_features(tmp: Path, T: int = 16, F: int = 3) -> Path:
    base = tmp / "features_gru_parity"
    base.mkdir(parents=True, exist_ok=True)
    rs = np.random.default_rng(11)
    dates = pd.date_range("2025-01-01", periods=T, freq="D").normalize()
    mkt = pd.DataFrame({
        "asof_date": dates,
        **{f"m{j}": rs.normal(0.0, 1.0, size=T).astype(float) for j in range(F)}
    })
    mkt.to_parquet(base / "market_features.parquet", index=False)

    def _mk(n: int) -> pd.DataFrame:
        di = rs.integers(0, T, size=n)
        side = rs.choice([-1.0, 1.0], size=n)
        lsz = rs.normal(0.0, 1.0, size=n)
        W = rs.normal(0.0, 0.25, size=F)
        y = []
        for i in range(n):
            mv = mkt[[f"m{j}" for j in range(F)]].iloc[int(di[i])].to_numpy()
            y.append(float(6.0 * float(mv @ W) + 2.0 * side[i] + 0.5 * lsz[i] + rs.normal(0, 0.05)))
        return pd.DataFrame({
            "trade_date": dates[di].astype("datetime64[ns]"),
            "isin": ["SIM" + str(i % 5) for i in range(n)],
            "side_sign": side.astype(float),
            "log_size": lsz.astype(float),
            "y_bps": np.array(y, dtype=float),
        })

    (base / "train.parquet").write_bytes(_mk(192).to_parquet(index=False))
    (base / "val.parquet").write_bytes(_mk(64).to_parquet(index=False))
    (base / "test.parquet").write_bytes(_mk(64).to_parquet(index=False))
    return base


def test_gru_scaling_matches_scorer(tmp_path: Path):
    # Train tiny GRU using baseline pipeline
    feature_dir = _make_baseline_features(tmp_path)
    out = tmp_path / "gru_out"
    cfg = GRUTrainConfig(
        feature_dir=str(feature_dir),
        outdir=str(out),
        device="cpu",
        epochs=2,
        lr=5e-3,
        batch_size=64,
        patience=0,
        seed=13,
        early_stopping=False,
        model=GRUModelConfig(hidden=32, layers=1, dropout=0.0, window=2, trade_cols=["side_sign","log_size"]),
    )
    train_gru(cfg)

    # Scorer predictions
    scorer = GRUScorer.from_dir(out, device="cpu")
    rows = [
        {"side": "buy", "size": 100000, "asof_date": None},
        {"side": "sell", "size": 25000,  "asof_date": None},
        {"side": "buy", "size": 0.0,     "asof_date": None},
    ]
    y_scorer = scorer.score_many(rows)

    # Direct model inference using exact same preprocessing as scorer (baseline artifacts)
    cfg_blob = json.loads((out / "config.json").read_text())
    mcfg = cfg_blob.get("model", {}) if isinstance(cfg_blob, dict) else {}
    trade_names = json.loads((out / "feature_names.json").read_text()).get("trade", ["side_sign","log_size"])
    st = json.loads((out / "scaler_trade.json").read_text())
    sm = json.loads((out / "scaler_market.json").read_text())
    t_mean = torch.tensor([float(v) for v in st.get("mean", [0.0]*len(trade_names))], dtype=torch.float32)
    t_std = torch.tensor([float(v) if float(v) > 0 else 1.0 for v in st.get("std", [1.0]*len(trade_names))], dtype=torch.float32)
    m_mean = torch.tensor([float(v) for v in sm.get("mean", [])], dtype=torch.float32)
    m_std = torch.tensor([float(v) if float(v) > 0 else 1.0 for v in sm.get("std", [])], dtype=torch.float32)

    # load market features from feature_dir
    mkt = pd.read_parquet(Path(cfg_blob["train"]["feature_dir"]) / "market_features.parquet")
    mkt = mkt.copy()
    mkt["asof_date"] = pd.to_datetime(mkt["asof_date"]).dt.normalize()
    mfeat = torch.as_tensor(mkt[[c for c in mkt.columns if c != "asof_date"]].astype(float).to_numpy(), dtype=torch.float32)

    ckpt = torch.load(out / "ckpt.pt", map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    from ptliq.training.gru_loop import GRURegressor
    model = GRURegressor(
        mkt_dim=int(mfeat.size(1)),
        trade_dim=int(len(trade_names)),
        hidden=int(mcfg.get("hidden", 64)),
        layers=int(mcfg.get("layers", 1)),
        dropout=float(mcfg.get("dropout", 0.0)),
    )
    model.load_state_dict(state_dict)
    model.eval()

    window = max(1, int(mcfg.get("window", 1)))
    n = len(rows)
    last_idx = int(mfeat.size(0) - 1)

    # market window ending at last_idx (as asof_date=None path in scorer)
    mseq = torch.zeros((n, window, int(mfeat.size(1))), dtype=torch.float32)
    for w in range(window):
        t = last_idx - (window - 1 - w)
        t = max(0, t)
        z = mfeat[t]
        denom_m = torch.where(m_std <= 0, torch.ones_like(m_std), m_std)
        z = (z - m_mean) / denom_m
        z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        mseq[:, w, :] = z

    # trade vectors
    side = torch.tensor([_side_sign(r.get("side")) for r in rows], dtype=torch.float32)
    lsz = torch.tensor([_log_size(r.get("size")) for r in rows], dtype=torch.float32)
    # align to trade_names order
    feats = []
    for name in trade_names:
        if name == "side_sign":
            feats.append(side)
        elif name == "log_size":
            feats.append(lsz)
        else:
            feats.append(torch.full((n,), float(0.0), dtype=torch.float32))
    trade_raw = torch.stack(feats, dim=1) if feats else torch.zeros((n, 0), dtype=torch.float32)
    denom_t = torch.where(t_std <= 0, torch.ones_like(t_std), t_std)
    trade = (trade_raw - t_mean) / denom_t
    trade = torch.nan_to_num(trade, nan=0.0, posinf=0.0, neginf=0.0)

    with torch.no_grad():
        y_direct = model(mseq, trade).detach().cpu().numpy().astype(np.float32)

    assert y_direct.shape == y_scorer.shape
    assert np.allclose(y_direct, y_scorer, rtol=1e-5, atol=1e-5), (
        f"GRU scorer vs direct mismatch:\nscorer={y_scorer}\ndirect={y_direct}"
    )
    assert np.all(np.abs(y_scorer) < 1000.0)


def test_mlp_scaling_matches_scorer(tmp_path: Path):
    # Train tiny MLP
    run_id = "exp_parity"
    features_base, _ = _make_mlp_features(tmp_path, run_id=run_id)
    models_dir = tmp_path / "models"
    mlp_train_main(
        features_dir=features_base.parent,
        run_id=run_id,
        models_dir=models_dir,
        device="cpu",
        max_epochs=2,
        batch_size=64,
        lr=1e-3,
        patience=2,
        seed=7,
        hidden="16,8",
        dropout=0.0,
        config=None,
        workdir=tmp_path,
        verbose=False,
    )

    outdir = models_dir / run_id

    scorer = MLPScorer.from_dir(outdir, device="cpu")

    # Build two rows with raw feature values
    rows = [
        {"f_one": 0.5, "f_two": -1.0},
        {"f_one": -0.1, "f_two": 0.4},
    ]
    y_scorer = scorer.score_many(rows)

    # Direct model inference
    feature_names = json.loads((outdir / "feature_names.json").read_text())
    sc = json.loads((outdir / "scaler.json").read_text())
    mean = np.array(sc.get("mean", [0.0, 0.0]), dtype=np.float32)
    std = np.array([v if float(v) > 0 else 1.0 for v in sc.get("std", [1.0, 1.0])], dtype=np.float32)

    # Build X in canonical order and standardize
    X_raw = []
    for r in rows:
        vec = []
        for name in feature_names:
            vec.append(float(r.get(name, 0.0)))
        X_raw.append(vec)
    X_raw = np.asarray(X_raw, dtype=np.float32)
    X = (X_raw - mean) / std
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # Load model
    from ptliq.model.baseline import MLPRegressor
    model_cfg_path = outdir / "model_config.json"
    if model_cfg_path.exists():
        mcfg = json.loads(model_cfg_path.read_text())
        hidden = list(map(int, mcfg.get("hidden", [64, 64])))
        dropout = float(mcfg.get("dropout", 0.0))
    else:
        # fallback for ancient artifacts
        hidden = [64, 64]
        dropout = 0.0
    model = MLPRegressor(in_dim=len(feature_names), hidden=hidden, dropout=dropout)
    ckpt = torch.load(outdir / "ckpt.pt", map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        y_direct = model(torch.from_numpy(X)).detach().cpu().numpy().astype(np.float32).reshape(-1)

    assert y_direct.shape == y_scorer.shape
    assert np.allclose(y_direct, y_scorer, rtol=1e-6, atol=1e-6), (
        f"MLP scorer vs direct mismatch:\nscorer={y_scorer}\ndirect={y_direct}"
    )
    assert np.all(np.abs(y_scorer) < 1000.0)
