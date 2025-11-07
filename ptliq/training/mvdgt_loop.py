# mvdgt_loop.py
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import torch

# Optional progress bar
try:  # pragma: no cover - tqdm is optional
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None  # type: ignore

# Optional TensorBoard
try:  # pragma: no cover - tensorboard is optional
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:  # pragma: no cover - tensorboard is optional
    SummaryWriter = None  # type: ignore

from ptliq.model.mv_dgt import MultiViewDGT
from ptliq.utils.attn_utils import enable_model_attn_capture, disable_model_attn_capture, log_attn_tb
from ptliq.utils.logging_utils import get_logger, setup_tb, safe_close_tb


@dataclass
class MVDGTModelConfig:
    hidden: int = 128
    heads: int = 2
    dropout: float = 0.10
    trade_dim: int = 2
    use_portfolio: bool = True
    # graph view names (order matters if masks are stored per view)
    views: List[str] = field(default_factory=lambda: ["struct", "port", "corr_global", "corr_local"])
    # optional per-sample portfolio head
    use_pf_head: bool = False
    pf_head_hidden: Optional[int] = None
    # runtime-computed fields persisted for exact reconstruction at inference time
    x_dim: Optional[int] = None
    mkt_dim: Optional[int] = None
    use_market: Optional[bool] = None


@dataclass
class MVDGTTrainConfig:
    # paths
    workdir: Path
    pyg_dir: Path
    outdir: Path
    # hparams
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    seed: int = 17
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # logging
    enable_tb: bool = True
    tb_log_dir: Optional[str] = None  # default to <outdir>/tb when None
    enable_tqdm: bool = True
    log_every: int = 50  # batch logging interval
    # attention TB logging
    enable_attn_tb: bool = False
    attn_log_every: int = 200  # batches
    # portfolio similarity aux loss (ID-agnostic)
    portfolio_sim_weight: float = 0.0
    portfolio_jitter_std: float = 0.05
    # model params (from YAML or CLI); defaults mirror previous hardcoded values
    model: MVDGTModelConfig = field(default_factory=MVDGTModelConfig)




class _DS(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        # side_sign, log_size may be absent in older datasets; default to 0.0
        side_sign = float(getattr(r, "side_sign", 0.0))
        log_size = float(getattr(r, "log_size", 0.0))
        return {
            "node_id": int(r.node_id),
            "date_idx": int(r.date_idx),
            "pf_gid": int(r.pf_gid),
            "side_sign": side_sign,
            "log_size": log_size,
            "y": float(r.y),
        }


def _collate(batch):
    out = {k: [] for k in batch[0].keys()}
    for b in batch:
        for k, v in b.items():
            out[k].append(v)
    return {k: torch.as_tensor(v) for k, v in out.items()}


def _metrics_bps(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict[str, float]:
    """
    Compute MSE, RMSE, and MAE in basis points (bps) given true and predicted tensors.
    y_true, y_pred: shapes (B,) or (B,1) assumed to be in bps.
    Returns float metrics in a dict with keys: mse_bps, rmse_bps, mae_bps.
    """
    err = (y_pred.view(-1) - y_true.view(-1)).float()
    mse = torch.mean(err * err)
    mae = torch.mean(err.abs())
    return {
        "mse_bps": float(mse.item()),
        "rmse_bps": float(torch.sqrt(mse + 1e-12).item()),
        "mae_bps": float(mae.item()),
    }


def train_mvdgt(cfg: MVDGTTrainConfig) -> dict:
    """
    Train the MV-DGT model using artifacts produced by the dataset builder.
    Save ALL model-related artifacts under <outdir> (checkpoint, configs, scaler, features, metrics, logs),
    while reading dataset artifacts from <workdir>.
    """
    _set_seed(cfg.seed)
    device = torch.device(cfg.device)

    workdir = Path(cfg.workdir)
    pyg_dir = Path(cfg.pyg_dir)
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- logging setup via util (log file under outdir/out by default)
    logger = get_logger("ptliq.training.mvdgt", outdir, filename="train_mvdgt.log")

    # --- load meta & artifacts; copy to outdir and augment if possible
    meta = copy_and_augment_meta(workdir, outdir, logger)

    # --- log training config early (centralized)
    log_and_write_training_config(cfg, outdir, logger)
    x, edge_index, edge_weight, view_masks = _load_pyg_and_view_masks(meta, workdir, device, outdir, logger)

    mkt_ctx, port_ctx = _load_context(meta.get("files", {}), device)

    # --- log context availability & stats
    _log_market_context_stats(mkt_ctx, logger)
    _log_portfolio_context_stats(port_ctx, logger)

    # datasets
    samp = pd.read_parquet(workdir / "samples.parquet")
    # Also persist a copy of samples.parquet into outdir for serving/UI convenience
    _copy_samples_to_outdir(samp, outdir, logger)

    tr = _DS(samp[samp["split"] == "train"])  # type: ignore[index]
    va = _DS(samp[samp["split"] == "val"])    # type: ignore[index]
    te = _DS(samp[samp["split"] == "test"])   # type: ignore[index]
    tr_loader = torch.utils.data.DataLoader(tr, batch_size=int(cfg.batch_size), shuffle=True, collate_fn=_collate)
    va_loader = torch.utils.data.DataLoader(va, batch_size=int(cfg.batch_size), shuffle=False, collate_fn=_collate)
    te_loader = torch.utils.data.DataLoader(te, batch_size=int(cfg.batch_size), shuffle=False, collate_fn=_collate)

    # --- Market preproc (mean/std + orientation sign) from TRAIN; persist and use during training
    mkt_mean_t, mkt_std_t, mkt_sign_t = _prepare_market_preproc_tensors(
        samp=samp, mkt_ctx=mkt_ctx, device=device, outdir=outdir, logger=logger
    )

    # --- Persist pack() compatibility artifacts: feature_names.json and scaler.json ---
    # For MV-DGT, per-trade numeric inputs are [side_sign, log_size]. Persist artifacts and build scaler tensors.
    tr_df = tr.df if hasattr(tr, "df") else None  # type: ignore[attr-defined]
    scaler_mean_t, scaler_std_t = _persist_trade_feature_artifacts(
        tr_df=tr_df,
        outdir=outdir,
        trade_dim=int(cfg.model.trade_dim),
        device=device,
    )

    # model
    mkt_dim = int(mkt_ctx["mkt_feat"].size(1)) if mkt_ctx is not None else 0

    # --- populate dataclass with runtime-computed fields and persist for exact reconstruction
    cfg.model.x_dim = int(x.size(1))
    cfg.model.mkt_dim = int(mkt_dim)
    cfg.model.use_market = bool(mkt_dim > 0)
    # persist JSON alongside training artifacts
    # centralized persistence of model config
    write_model_config_json(cfg.model, outdir, logger)

    model = MultiViewDGT(
        x_dim=int(cfg.model.x_dim or x.size(1)),
        hidden=int(cfg.model.hidden),
        heads=int(cfg.model.heads),
        dropout=float(cfg.model.dropout),
        view_masks=view_masks,
        edge_index=edge_index,
        edge_weight=edge_weight,
        mkt_dim=int(cfg.model.mkt_dim or mkt_dim),
        use_portfolio=bool(cfg.model.use_portfolio),
        use_market=bool(cfg.model.use_market if cfg.model.use_market is not None else (mkt_dim > 0)),
        trade_dim=int(cfg.model.trade_dim),
        view_names=list(cfg.model.views),
        use_pf_head=bool(getattr(cfg.model, "use_pf_head", False)),
        pf_head_hidden=getattr(cfg.model, "pf_head_hidden", None),
    ).to(device)

    # --- runtime sanity prints for correlation gate and edge counts (centralized)
    log_edge_counts_and_corr_gate(model, logger)

    # standardized residuals stats from train split
    y_tr = pd.read_parquet(workdir / "samples.parquet")
    y_tr = y_tr[y_tr["split"] == "train"]["y"].astype(float).to_numpy()
    y_mu  = torch.tensor(float(y_tr.mean()), device=device)
    y_std = torch.tensor(float(max(y_tr.std(), 1e-6)), device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=float(cfg.lr),
        steps_per_epoch=len(tr_loader),
        epochs=int(cfg.epochs),
        pct_start=0.1,
        anneal_strategy="cos",
    )
    loss_fn = torch.nn.SmoothL1Loss(reduction="none")

    # TensorBoard writer (optional)
    writer = setup_tb(enable_tb=cfg.enable_tb, workdir=outdir, tb_log_dir=cfg.tb_log_dir)

    # progress bar util
    def _tqdm(iterable, total=None, **kwargs):
        if not (cfg.enable_tqdm and tqdm is not None):
            return iterable
        return tqdm(iterable, total=total, **kwargs)

    global_step = 0

    def _run(loader, train_flag=False, epoch: int = 0, phase: str = "train"):
        nonlocal global_step
        model.train(train_flag)
        tot_mse_bps, tot_mae_bps, n = 0.0, 0.0, 0
        it = _tqdm(loader, total=len(loader), leave=False, desc=f"{phase} ep{epoch:03d}")
        with torch.set_grad_enabled(train_flag):
            for i, batch in enumerate(it, start=1):
                anchor = batch["node_id"].long().to(device)
                pf_gid = batch["pf_gid"].long().to(device)
                y = batch["y"].float().to(device)
                # market row pick (with optional z-score + orientation sign)
                mkt = _standardize_market_batch(mkt_ctx, batch, device, mkt_mean_t, mkt_std_t, mkt_sign_t)
                # trade features (standardized using train-split scaler tensors)
                trade = _standardize_trade_batch(batch, device, scaler_mean_t, scaler_std_t)

                # optionally capture attention stats this step
                capture_attn = bool(
                    train_flag and (writer is not None) and cfg.enable_attn_tb and (
                        global_step % max(1, int(cfg.attn_log_every)) == 0
                    )
                )
                if capture_attn:
                    enable_model_attn_capture(model)
                # Build dynamic per-batch portfolio context (permutation-invariant)
                dyn_port_ctx = _build_dyn_port_ctx_from_batch(
                    anchor=anchor,
                    pf_gid=pf_gid,
                    side_sign=batch["side_sign"],
                    log_size=batch["log_size"],
                )
                need_aux = bool(cfg.portfolio_sim_weight > 0.0)
                out = model(x, anchor_idx=anchor, market_feat=mkt, pf_gid=pf_gid, port_ctx=(dyn_port_ctx or port_ctx), trade_feat=trade, return_aux=need_aux)
                if need_aux:
                    yhat, aux = out
                else:
                    yhat = out
                    aux = {}
                # if captured, push attention summaries to TB
                if capture_attn:
                    stats = getattr(model, "_attn_stats", {}) or {}
                    log_attn_tb(writer, stats, global_step, prefix="attn/")
                    disable_model_attn_capture(model)

                # compute loss (standardized error + optional similarity)
                loss, loss_comp = _compute_train_loss(yhat, y, y_std, pf_gid, loss_fn, aux, cfg)

                if train_flag:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    sched.step()
                bsz = int(y.size(0))
                # compute and accumulate true bps metrics
                m = _metrics_bps(y_true=y.detach(), y_pred=yhat.detach())
                tot_mse_bps += m["mse_bps"] * bsz
                tot_mae_bps += m["mae_bps"] * bsz
                n += bsz

                # per-batch logging (train only)
                if train_flag:
                    _log_batch_tb(writer, loss, m, global_step)
                    if cfg.enable_tqdm and tqdm is not None:
                        it.set_postfix({"loss": f"{float(loss.item()):.4f}"})
                    global_step += 1
        avg_mse = tot_mse_bps / max(1, n)
        avg_rmse = avg_mse ** 0.5
        avg_mae = tot_mae_bps / max(1, n)
        _log_epoch_tb(writer, phase, avg_mse, avg_rmse, avg_mae, epoch)        
        return avg_mse, avg_mae

    best = 1e9
    best_state = None
    history = []
    num_epochs = int(cfg.epochs)
    ep_iter = _tqdm(range(1, num_epochs + 1), total=num_epochs, desc="epochs")
    for ep in ep_iter:
        tr_loss, tr_mae = _run(tr_loader, train_flag=True, epoch=ep, phase="train")
        va_loss, va_mae = _run(va_loader, train_flag=False, epoch=ep, phase="val")
        if writer is not None:
            writer.add_scalar("lr", float(opt.param_groups[0].get("lr", cfg.lr)), ep)
            # log corr gate value each epoch for monitoring
            try:
                writer.add_scalar("model/corr_gate", float(torch.sigmoid(model.corr_gate).item()), ep)
            except Exception:
                pass
        if va_loss < best:
            best = va_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        if cfg.enable_tqdm and tqdm is not None:
            ep_iter.set_postfix({"val_mse": f"{va_loss:.4f}", "train_mse": f"{tr_loss:.4f}"})
        logger.info(f"epoch {ep:03d} train_mae={tr_mae:.4f} train_rmse={(tr_loss ** 0.5):.4f} val_mae={va_mae:.4f} val_rmse={(va_loss ** 0.5):.4f}")
        history.append({"epoch": ep, "train_mse": float(tr_loss), "train_mae": float(tr_mae), "val_mse": float(va_loss), "val_mae": float(va_mae)})
    if best_state is not None:
        model.load_state_dict(best_state)
    te_loss, te_mae = _run(te_loader, train_flag=False, epoch=num_epochs + 1, phase="test")
    te_rmse = float(te_loss ** 0.5)
    logger.info(f"[TEST] RMSE = {te_rmse:.4f} | MAE = {te_mae:.4f} | MSE = {te_loss:.4f}")

    # save (centralized)
    save_checkpoint(model, meta, cfg.model, outdir, logger)

    safe_close_tb(writer)

    return {"val_mse": float(best), "test_mse": float(te_loss), "test_rmse": te_rmse, "test_mae": float(te_mae), "history": history}


# ---------------- Helper utilities for readability (no behavior change) ----------------

def _set_seed(seed: int) -> None:
    """Set RNG seeds for torch, numpy, and random. Local implementation (no external deps)."""
    torch.manual_seed(int(seed))
    try:
        import random
        import numpy as np  # type: ignore
        random.seed(int(seed))
        np.random.seed(int(seed))
    except Exception:
        pass


def _load_context(meta_files: Dict[str, Any], device: torch.device) -> tuple[Optional[dict], Optional[dict]]:
    # market / portfolio contexts (optional)
    mkt_ctx = None
    mkt_path_str = meta_files.get("market_context")
    if mkt_path_str:
        mkt_path = Path(mkt_path_str)
        if mkt_path.is_file():
            mkt_ctx = torch.load(mkt_path, map_location=device)

    port_ctx = None
    port_path_str = meta_files.get("portfolio_context")
    if port_path_str:
        port_path = Path(port_path_str)
        if port_path.is_file():
            port_ctx = torch.load(port_path, map_location=device)
    return mkt_ctx, port_ctx


def _compute_market_preproc(samples_df: pd.DataFrame, mkt_ctx: Optional[dict], logger) -> Optional[dict]:
    """
    Compute mean/std for market features and a sign calibration based on a small ridge fit of y on z-scored mkt_feat.
    Uses TRAIN rows only. Returns {"mean": list, "std": list, "sign": float} or None.
    """
    try:
        if (mkt_ctx is None) or (not isinstance(mkt_ctx, dict)) or ("mkt_feat" not in mkt_ctx):
            return None
        mf = mkt_ctx["mkt_feat"].float()  # [T, F]
        m_mean = mf.mean(dim=0)
        m_std = mf.std(dim=0, unbiased=False).clamp_min(1e-6)
        df = samples_df[samples_df["split"] == "train"]["date_idx"].astype(int)
        ycol = samples_df[samples_df["split"] == "train"]["y"].astype(float)
        if len(df) < 10:
            return {"mean": m_mean.detach().cpu().tolist(), "std": m_std.detach().cpu().tolist(), "sign": 1.0}
        idx = torch.as_tensor(df.to_numpy(), dtype=torch.long)
        M = mf.index_select(0, idx).float()  # [N, F]
        y = torch.as_tensor(ycol.to_numpy(), dtype=torch.float32).view(-1, 1)
        Mz = (M - m_mean) / m_std
        lam = 1e-3
        ATA = Mz.T @ Mz + lam * torch.eye(Mz.size(1))
        b = torch.linalg.solve(ATA, Mz.T @ y)  # [F,1]
        fit = (Mz @ b).view(-1)
        try:
            fit_np = fit.detach().cpu().numpy()
            y_np = y.view(-1).detach().cpu().numpy()
            corr = float(np.corrcoef(fit_np, y_np)[0, 1])
            if not (corr == corr):  # NaN guard
                corr = 0.0
        except Exception:
            corr = 0.0
        sgn = 1.0 if corr >= 0.0 else -1.0
        return {"mean": m_mean.detach().cpu().tolist(), "std": m_std.detach().cpu().tolist(), "sign": float(sgn)}
    except Exception as e:
        try:
            logger.warning(f"market preproc computation failed: {e}")
        except Exception:
            pass
        return None

def _standardize_market_batch(mkt_ctx: Optional[dict], batch: dict, device: torch.device, mkt_mean_t: Optional[torch.Tensor], mkt_std_t: Optional[torch.Tensor], mkt_sign_t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if mkt_ctx is None:
        return None
    di = batch["date_idx"].long().clamp_min(0).to(device)
    mkt_raw = mkt_ctx["mkt_feat"].index_select(0, di)
    if (mkt_mean_t is not None) and (mkt_std_t is not None):
        denom_m = torch.where(mkt_std_t <= 0, torch.ones_like(mkt_std_t), mkt_std_t)
        mkt = (mkt_raw - mkt_mean_t) / denom_m
        mkt = torch.nan_to_num(mkt, nan=0.0, posinf=0.0, neginf=0.0)
        if mkt_sign_t is not None:
            mkt = mkt * mkt_sign_t
    else:
        mkt = mkt_raw
    return mkt


def _standardize_trade_batch(batch: dict, device: torch.device, scaler_mean_t: torch.Tensor, scaler_std_t: torch.Tensor) -> torch.Tensor:
    trade_raw = torch.stack([
        batch["side_sign"].float().to(device),
        batch["log_size"].float().to(device),
    ], dim=1)
    denom = torch.where(scaler_std_t <= 0, torch.ones_like(scaler_std_t), scaler_std_t)
    trade = (trade_raw - scaler_mean_t) / denom
    trade = torch.nan_to_num(trade, nan=0.0, posinf=0.0, neginf=0.0)
    return trade


def _build_dyn_port_ctx_from_batch(anchor: torch.Tensor, pf_gid: torch.Tensor, side_sign: torch.Tensor, log_size: torch.Tensor) -> Optional[dict]:
    """Build dynamic per-batch portfolio context using permutation-invariant weights.
    abs ∝ expm1(log_size.clamp_min(0)); signed ∝ sign(side) * abs; both normalized per pf_gid.
    """
    device = anchor.device
    gids = pf_gid.view(-1).tolist()
    with torch.no_grad():
        abs_sizes = torch.expm1(log_size.float().to(device).clamp_min(0.0))
        sides = side_sign.float().to(device)
        nodes = anchor.long()
        groups: dict[int, list[int]] = {}
        for i, g in enumerate(gids):
            if int(g) < 0:
                continue
            groups.setdefault(int(g), []).append(i)
        if len(groups) == 0:
            return None
        port_nodes: list[int] = []
        w_abs: list[float] = []
        w_sgn: list[float] = []
        port_len: list[int] = []
        for _g, idxs_g in groups.items():
            a = abs_sizes[idxs_g]
            s = torch.sign(sides[idxs_g]) * a
            denom = torch.clamp(a.sum(), min=1.0)
            port_len.append(len(idxs_g))
            port_nodes.extend(nodes[idxs_g].tolist())
            w_abs.extend((a / denom).tolist())
            w_sgn.extend((s / denom).tolist())
        return {
            "port_nodes_flat": torch.tensor(port_nodes, dtype=torch.long, device=device),
            "port_w_abs_flat": torch.tensor(w_abs, dtype=torch.float32, device=device),
            "port_w_signed_flat": torch.tensor(w_sgn, dtype=torch.float32, device=device),
            "port_len": torch.tensor(port_len, dtype=torch.long, device=device),
        }


def _portfolio_similarity_loss(z_pf: torch.Tensor, pf_gid: torch.Tensor) -> torch.Tensor:
    """Within-portfolio variance loss. Returns a scalar tensor on z_pf's device."""
    device = z_pf.device
    gids = pf_gid.view(-1).tolist()
    sim_loss = torch.tensor(0.0, device=device)
    cnt = 0
    for g in sorted(set(gids)):
        if g is None or int(g) < 0:
            continue
        idxs_g = [i for i, gg in enumerate(gids) if gg == g]
        if len(idxs_g) <= 1:
            continue
        Zg = z_pf.index_select(0, torch.tensor(idxs_g, device=device))
        mu_g = Zg.mean(dim=0, keepdim=True)
        sim_loss = sim_loss + torch.mean((Zg - mu_g).pow(2))
        cnt += 1
    if cnt > 0:
        sim_loss = sim_loss / float(cnt)
    return sim_loss


# ---------------- helper utilities  ----------------

def _load_pyg_and_view_masks(meta: dict, workdir: Path, device: torch.device, outdir: Path, logger) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], dict]:
    """Load PyG graph tensors and view masks, copy masks to outdir, and move to device.
    Returns (x, edge_index, edge_weight, view_masks_on_device).
    """
    data = torch.load(meta["files"]["pyg_graph"], map_location="cpu", weights_only=False)
    view_masks = torch.load(workdir / "view_masks.pt", map_location="cpu", weights_only=False)
    # also persist a copy to outdir for DGTScorer compatibility
    try:
        vm_cpu = {k: v.detach().cpu() for k, v in view_masks.items()}
        torch.save(vm_cpu, outdir / "view_masks.pt")
    except Exception:
        logger.warning("failed to copy view_masks.pt to outdir")
    # inputs: node features (from PyG), edge stuff for building the model
    x = data.x.float().to(device)
    edge_index = data.edge_index.to(device)
    edge_weight = data.edge_weight.to(device) if hasattr(data, "edge_weight") else None
    for k in list(view_masks.keys()):
        view_masks[k] = view_masks[k].to(device)
    return x, edge_index, edge_weight, view_masks


def _log_market_context_stats(mkt_ctx: Optional[dict], logger) -> None:
    """Log availability and summary stats for market context."""
    try:
        if mkt_ctx is not None and isinstance(mkt_ctx, dict) and ("mkt_feat" in mkt_ctx):
            mf = mkt_ctx["mkt_feat"]
            logger.info(f"market_context: available | mkt_feat shape={tuple(mf.shape)} dtype={mf.dtype} device={mf.device}")
            try:
                mv = float(mf.float().mean().item())
                sd = float(mf.float().std(unbiased=False).item())
                logger.info(f"market_context: mean={mv:.6f} std={sd:.6f}")
            except Exception:
                pass
        else:
            logger.info("market_context: not available")
    except Exception:
        logger.warning("failed to log market context stats")


def _log_portfolio_context_stats(port_ctx: Optional[dict], logger) -> None:
    """Log availability and summary stats for portfolio context."""
    try:
        if port_ctx is not None and isinstance(port_ctx, dict):
            required = {"port_nodes_flat","port_w_signed_flat","port_len"}
            missing = required.difference(set(port_ctx.keys()))
            if missing:
                logger.info(f"portfolio_context: available but missing keys={sorted(missing)}")
            nodes_flat = port_ctx.get("port_nodes_flat")
            lens = port_ctx.get("port_len")
            if nodes_flat is not None and lens is not None:
                L = int(nodes_flat.numel())
                G = int(lens.numel())
                avg = float(lens.float().mean().item()) if G > 0 else 0.0
                mn = int(lens.min().item()) if G > 0 else 0
                mx = int(lens.max().item()) if G > 0 else 0
                logger.info(f"portfolio_context: available | groups={G} line_items={L} avg_len={avg:.2f} min_len={mn} max_len={mx}")
            else:
                logger.info("portfolio_context: available | stats unavailable (missing tensors)")
        else:
            logger.info("portfolio_context: not available")
    except Exception:
        logger.warning("failed to log portfolio context stats")


def _copy_samples_to_outdir(samp: pd.DataFrame, outdir: Path, logger) -> None:
    """Persist a copy of samples.parquet into outdir for serving/UI convenience."""
    try:
        out_samp_path = outdir / "samples.parquet"
        samp.to_parquet(out_samp_path, index=False)
        logger.info(f"copied samples.parquet to outdir: {out_samp_path}")
    except Exception as e:
        logger.warning(f"failed to copy samples.parquet to outdir: {e}")


def _persist_trade_feature_artifacts(tr_df: Optional[pd.DataFrame], outdir: Path, trade_dim: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Write feature_names.json and scaler.json; return scaler tensors on device.
    Expects trade_dim to match number of features (2: side_sign, log_size).
    """
    # Defaults in case artifact writing fails
    scaler_mean_t = torch.tensor([0.0, 0.0], dtype=torch.float32, device=device)
    scaler_std_t = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)
    try:
        feature_names = ["side_sign", "log_size"]
        if tr_df is not None and all(c in tr_df.columns for c in feature_names):
            mean = [float(tr_df[c].astype(float).mean()) for c in feature_names]
            std = [float(tr_df[c].astype(float).std(ddof=0)) for c in feature_names]
            std = [1.0 if (not (s > 0.0)) or (s != s) else float(s) for s in std]
            mean = [0.0 if (m != m) else float(m) for m in mean]
        else:
            mean = [0.0 for _ in feature_names]
            std = [1.0 for _ in feature_names]
        (outdir / "feature_names.json").write_text(json.dumps(feature_names, indent=2))
        (outdir / "scaler.json").write_text(json.dumps({"mean": mean, "std": std}, indent=2))
        scaler_mean_t = torch.tensor(mean, dtype=torch.float32, device=device)
        scaler_std_t = torch.tensor([1.0 if (s is None or not (s > 0.0)) else float(s) for s in std], dtype=torch.float32, device=device)
        assert len(feature_names) == int(trade_dim), \
            f"trade_dim={trade_dim} but feature_names has {len(feature_names)} items"
    except Exception as e:
        # keep defaults if anything goes wrong
        try:
            import logging as _lg
            _lg.getLogger("ptliq.training.mvdgt").warning(f"failed to write feature_names/scaler artifacts: {e}")
        except Exception:
            pass
    return scaler_mean_t, scaler_std_t

def _standardize_market_batch(mkt_ctx: Optional[dict], batch: dict, device: torch.device, mkt_mean_t: Optional[torch.Tensor], mkt_std_t: Optional[torch.Tensor], mkt_sign_t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if mkt_ctx is None:
        return None
    di = batch["date_idx"].long().clamp_min(0).to(device)
    mkt_raw = mkt_ctx["mkt_feat"].index_select(0, di)
    if (mkt_mean_t is not None) and (mkt_std_t is not None):
        denom_m = torch.where(mkt_std_t <= 0, torch.ones_like(mkt_std_t), mkt_std_t)
        mkt = (mkt_raw - mkt_mean_t) / denom_m
        mkt = torch.nan_to_num(mkt, nan=0.0, posinf=0.0, neginf=0.0)
        if mkt_sign_t is not None:
            mkt = mkt * mkt_sign_t
    else:
        mkt = mkt_raw
    return mkt


def _standardize_trade_batch(batch: dict, device: torch.device, scaler_mean_t: torch.Tensor, scaler_std_t: torch.Tensor) -> torch.Tensor:
    trade_raw = torch.stack([
        batch["side_sign"].float().to(device),
        batch["log_size"].float().to(device),
    ], dim=1)
    denom = torch.where(scaler_std_t <= 0, torch.ones_like(scaler_std_t), scaler_std_t)
    trade = (trade_raw - scaler_mean_t) / denom
    trade = torch.nan_to_num(trade, nan=0.0, posinf=0.0, neginf=0.0)
    return trade


def _build_dyn_port_ctx_from_batch(anchor: torch.Tensor, pf_gid: torch.Tensor, side_sign: torch.Tensor, log_size: torch.Tensor) -> Optional[dict]:
    """Build dynamic per-batch portfolio context using permutation-invariant weights.
    abs ∝ expm1(log_size.clamp_min(0)); signed ∝ sign(side) * abs; both normalized per pf_gid.
    """
    device = anchor.device
    gids = pf_gid.view(-1).tolist()
    with torch.no_grad():
        abs_sizes = torch.expm1(log_size.float().to(device).clamp_min(0.0))
        sides = side_sign.float().to(device)
        nodes = anchor.long()
        groups: dict[int, list[int]] = {}
        for i, g in enumerate(gids):
            if int(g) < 0:
                continue
            groups.setdefault(int(g), []).append(i)
        if len(groups) == 0:
            return None
        port_nodes: list[int] = []
        w_abs: list[float] = []
        w_sgn: list[float] = []
        port_len: list[int] = []
        for _g, idxs_g in groups.items():
            a = abs_sizes[idxs_g]
            s = torch.sign(sides[idxs_g]) * a
            denom = torch.clamp(a.sum(), min=1.0)
            port_len.append(len(idxs_g))
            port_nodes.extend(nodes[idxs_g].tolist())
            w_abs.extend((a / denom).tolist())
            w_sgn.extend((s / denom).tolist())
        return {
            "port_nodes_flat": torch.tensor(port_nodes, dtype=torch.long, device=device),
            "port_w_abs_flat": torch.tensor(w_abs, dtype=torch.float32, device=device),
            "port_w_signed_flat": torch.tensor(w_sgn, dtype=torch.float32, device=device),
            "port_len": torch.tensor(port_len, dtype=torch.long, device=device),
        }


def _portfolio_similarity_loss(z_pf: torch.Tensor, pf_gid: torch.Tensor) -> torch.Tensor:
    """Within-portfolio variance loss. Returns a scalar tensor on z_pf's device."""
    device = z_pf.device
    gids = pf_gid.view(-1).tolist()
    sim_loss = torch.tensor(0.0, device=device)
    cnt = 0
    for g in sorted(set(gids)):
        if g is None or int(g) < 0:
            continue
        idxs_g = [i for i, gg in enumerate(gids) if gg == g]
        if len(idxs_g) <= 1:
            continue
        Zg = z_pf.index_select(0, torch.tensor(idxs_g, device=device))
        mu_g = Zg.mean(dim=0, keepdim=True)
        sim_loss = sim_loss + torch.mean((Zg - mu_g).pow(2))
        cnt += 1
    if cnt > 0:
        sim_loss = sim_loss / float(cnt)
    return sim_loss


def copy_and_augment_meta(workdir: Path, outdir: Path, logger) -> Dict[str, Any]:
    """Copy mvdgt_meta.json from workdir to outdir and augment with market_index if adjacent."""
    meta_path = workdir / "mvdgt_meta.json"
    meta = json.loads(meta_path.read_text())
    try:
        (outdir / "mvdgt_meta.json").write_text(json.dumps(meta, indent=2))
        try:
            meta_out = dict(meta)
            files = dict(meta_out.get("files", {}))
            pyg_graph_path = Path(files.get("pyg_graph", ""))
            mkt_index_guess = pyg_graph_path.parent / "market_index.parquet"
            if mkt_index_guess.exists():
                files["market_index"] = str(mkt_index_guess)
                meta_out["files"] = files
                (outdir / "mvdgt_meta.json").write_text(json.dumps(meta_out, indent=2))
        except Exception:
            logger.warning("failed to augment mvdgt_meta.json with market_index")
    except Exception:
        logger.warning("failed to copy mvdgt_meta.json to outdir")
    return meta


def log_and_write_training_config(cfg, outdir: Path, logger) -> None:
    """Log full training config and persist to outdir/train_config.json."""
    try:
        cfg_dict = asdict(cfg)
        for k, v in list(cfg_dict.items()):
            if isinstance(v, Path):
                cfg_dict[k] = str(v)
        logger.info("training_config=" + json.dumps(cfg_dict, ensure_ascii=False))
        try:
            (outdir / "train_config.json").write_text(json.dumps(cfg_dict, indent=2))
        except Exception:
            pass
    except Exception:
        logger.warning("failed to log training config")


def write_model_config_json(model_cfg, outdir: Path, logger=None) -> None:
    try:
        (outdir / "model_config.json").write_text(json.dumps(asdict(model_cfg), indent=2))
    except Exception:
        if logger is not None:
            logger.warning("failed to persist model_config.json")


def log_edge_counts_and_corr_gate(model, logger) -> None:
    try:
        corr_gate_val = float(torch.sigmoid(model.corr_gate).item())
        logger.info(f"corr_gate = {corr_gate_val:.6f}")
    except Exception:
        logger.warning("corr_gate = <unavailable>")
    try:
        def _mask_count(name: str):
            if hasattr(model, f"mask_{name}"):
                return int(getattr(model, f"mask_{name}").sum().item())
            return None
        n_struct = _mask_count("struct")
        n_port = _mask_count("port")
        n_corr_global = _mask_count("corr_global")
        n_corr_local = _mask_count("corr_local")
        logger.info(f"num struct      edges = {n_struct if n_struct is not None else '<mask missing>'}")
        logger.info(f"num port        edges = {n_port if n_port is not None else '<mask missing>'}")
        logger.info(f"num corr_global edges = {n_corr_global if n_corr_global is not None else '<mask missing>'}")
        logger.info(f"num corr_local  edges = {n_corr_local if n_corr_local is not None else '<mask missing>'}")
    except Exception:
        logger.warning("<failed to compute edge counts>")


def save_checkpoint(model, meta: Dict[str, Any], model_cfg, outdir: Path, logger) -> None:
    ckpt_obj = {
        "state_dict": model.state_dict(),
        "meta": meta,
        "model_config": asdict(model_cfg),
    }
    try:
        torch.save(ckpt_obj, outdir / "ckpt.pt")
    except Exception as e:
        logger.warning(f"failed to save checkpoint to ckpt.pt: {e}")


# ---------------- New helpers extracted per request ----------------

def _prepare_market_preproc_tensors(samp: pd.DataFrame, mkt_ctx: Optional[dict], device: torch.device, outdir: Path, logger):
    mkt_mean_t = None
    mkt_std_t = None
    mkt_sign_t = None
    try:
        pre = _compute_market_preproc(samp, mkt_ctx, logger)
        if pre is not None:
            try:
                (outdir / "market_preproc.json").write_text(json.dumps(pre, indent=2))
            except Exception:
                logger.warning("failed to write market_preproc.json")
            try:
                mkt_mean_t = torch.tensor(pre["mean"], dtype=torch.float32, device=device)
                mkt_std_vals = [float(v) if (v is not None and float(v) > 0.0) else 1.0 for v in pre["std"]]
                mkt_std_t = torch.tensor(mkt_std_vals, dtype=torch.float32, device=device)
                mkt_sign_t = torch.tensor(float(pre.get("sign", 1.0)), dtype=torch.float32, device=device)
            except Exception as e:
                logger.warning(f"failed to build market preproc tensors: {e}")
    except Exception as e:
        logger.warning(f"market preproc computation failed: {e}")
    return mkt_mean_t, mkt_std_t, mkt_sign_t


def _compute_train_loss(yhat: torch.Tensor, y: torch.Tensor, y_std: torch.Tensor, pf_gid: torch.Tensor, loss_fn: torch.nn.Module, aux: dict, cfg: MVDGTTrainConfig) -> tuple[torch.Tensor, dict]:
    err = yhat - y
    err_z = err / y_std
    w = torch.ones_like(err_z)
    w = w * (1.0 + 1.0 * (pf_gid >= 0).float())  # double-weight portfolio trades
    loss_vec = loss_fn(err_z, torch.zeros_like(err_z))
    loss = (w * loss_vec).mean()
    comps = {"base": loss.detach()}
    need_aux = bool(cfg.portfolio_sim_weight > 0.0)
    if need_aux and ("z_pf" in aux):
        z_pf = aux["z_pf"]
        sim_loss = _portfolio_similarity_loss(z_pf, pf_gid)
        loss = loss + float(cfg.portfolio_sim_weight) * sim_loss
        comps["sim"] = sim_loss.detach()
    return loss, comps


def _log_batch_tb(writer, loss: torch.Tensor | None, metrics: dict, global_step: int) -> None:
    if writer is None:
        return
    try:
        writer.add_scalar("train/batch_loss", float(loss.item()) if loss is not None else float("nan"), global_step)
        writer.add_scalar("train/mse_bps", metrics.get("mse_bps", float("nan")), global_step)
        writer.add_scalar("train/rmse_bps", metrics.get("rmse_bps", float("nan")), global_step)
        writer.add_scalar("train/mae_bps", metrics.get("mae_bps", float("nan")), global_step)
    except Exception:
        pass


def _log_epoch_tb(writer, phase: str, avg_mse: float, avg_rmse: float, avg_mae: float, epoch: int) -> None:
    if writer is None:
        return
    try:
        writer.add_scalar(f"{phase}/mse_bps", float(avg_mse), epoch)
        writer.add_scalar(f"{phase}/rmse_bps", float(avg_rmse), epoch)
        writer.add_scalar(f"{phase}/mae_bps", float(avg_mae), epoch)
    except Exception:
        pass
