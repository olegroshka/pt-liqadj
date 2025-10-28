from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

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


@dataclass
class MVDGTTrainConfig:
    # paths
    workdir: Path
    pyg_dir: Path
    # hparams
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    seed: int = 17
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # logging
    enable_tb: bool = True
    tb_log_dir: Optional[str] = None  # default to <workdir>/tb when None
    enable_tqdm: bool = True
    log_every: int = 50  # batch logging interval


def _set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    try:
        import random, numpy as np  # type: ignore
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


def train_mvdgt(cfg: MVDGTTrainConfig) -> dict:
    """
    Train the MV-DGT model using artifacts produced by the dataset builder.
    Saves a checkpoint under <workdir>/mv_dgt_ckpt.pt and returns metrics.
    Adds optional TensorBoard logging and tqdm progress bars similar to GAT loop.
    """
    _set_seed(cfg.seed)
    device = torch.device(cfg.device)

    workdir = Path(cfg.workdir)
    pyg_dir = Path(cfg.pyg_dir)

    # --- load meta & artifacts
    meta = json.loads((workdir / "mvdgt_meta.json").read_text())
    data = torch.load(meta["files"]["pyg_graph"], map_location="cpu", weights_only=False)
    view_masks = torch.load(workdir / "view_masks.pt", map_location="cpu", weights_only=False)

    # inputs: node features (from PyG), edge stuff for building the model
    x = data.x.float().to(device)
    edge_index = data.edge_index.to(device)
    edge_weight = data.edge_weight.to(device) if hasattr(data, "edge_weight") else None
    for k in list(view_masks.keys()):
        view_masks[k] = view_masks[k].to(device)

    mkt_ctx, port_ctx = _load_context(meta.get("files", {}), device)

    # datasets
    samp = pd.read_parquet(workdir / "samples.parquet")

    tr = _DS(samp[samp["split"] == "train"])  # type: ignore[index]
    va = _DS(samp[samp["split"] == "val"])    # type: ignore[index]
    te = _DS(samp[samp["split"] == "test"])   # type: ignore[index]
    tr_loader = torch.utils.data.DataLoader(tr, batch_size=int(cfg.batch_size), shuffle=True, collate_fn=_collate)
    va_loader = torch.utils.data.DataLoader(va, batch_size=int(cfg.batch_size), shuffle=False, collate_fn=_collate)
    te_loader = torch.utils.data.DataLoader(te, batch_size=int(cfg.batch_size), shuffle=False, collate_fn=_collate)

    # model
    mkt_dim = int(mkt_ctx["mkt_feat"].size(1)) if mkt_ctx is not None else 0
    model = MultiViewDGT(
        x_dim=int(x.size(1)),
        hidden=128,
        heads=2,
        dropout=0.1,
        view_masks=view_masks,
        edge_index=edge_index,
        edge_weight=edge_weight,
        mkt_dim=mkt_dim,
        use_portfolio=True,
        use_market=(mkt_dim > 0),
        trade_dim=2,
    ).to(device)

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
    writer = None
    if cfg.enable_tb and SummaryWriter is not None:
        tb_dir = Path(cfg.tb_log_dir) if cfg.tb_log_dir else (workdir / "tb")
        try:
            tb_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=str(tb_dir))
        except Exception:
            writer = None  # fallback silently

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
                # market row pick
                if mkt_ctx is not None:
                    di = batch["date_idx"].long().clamp_min(0).to(device)
                    mkt = mkt_ctx["mkt_feat"].index_select(0, di)
                else:
                    mkt = None
                # trade features
                trade = torch.stack([
                    batch["side_sign"].float().to(device),
                    batch["log_size"].float().to(device),
                ], dim=1)

                yhat = model(x, anchor_idx=anchor, market_feat=mkt, pf_gid=pf_gid, port_ctx=port_ctx, trade_feat=trade)

                # standardized error; robust loss with portfolio weighting
                err = yhat - y
                err_z = err / y_std
                w = torch.ones_like(err_z)
                w = w * (1.0 + 1.0 * (pf_gid >= 0).float())  # double-weight portfolio trades
                loss_vec = loss_fn(err_z, torch.zeros_like(err_z))
                loss = (w * loss_vec).mean()

                if train_flag:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    sched.step()
                bsz = int(y.size(0))
                # accumulate true bps metrics
                batch_mse = float((err.detach() ** 2).mean().item())
                batch_mae = float(torch.abs(err.detach()).mean().item())
                tot_mse_bps += batch_mse * bsz
                tot_mae_bps += batch_mae * bsz
                n += bsz

                # per-batch logging
                if train_flag:
                    if writer is not None:
                        writer.add_scalar("train/batch_loss", float(loss.item()), global_step)
                    if cfg.enable_tqdm and tqdm is not None:
                        it.set_postfix({"loss": f"{float(loss.item()):.4f}"})
                    global_step += 1
        avg_mse = tot_mse_bps / max(1, n)
        avg_rmse = avg_mse ** 0.5
        avg_mae = tot_mae_bps / max(1, n)
        if writer is not None:
            writer.add_scalar(f"{phase}/rmse_bps", float(avg_rmse), epoch)
            writer.add_scalar(f"{phase}/mae_bps", float(avg_mae), epoch)
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
        if va_loss < best:
            best = va_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        if cfg.enable_tqdm and tqdm is not None:
            ep_iter.set_postfix({"val_mse": f"{va_loss:.4f}", "train_mse": f"{tr_loss:.4f}"})
        print(f"epoch {ep:03d} train_mse={tr_loss:.4f} val_mse={va_loss:.4f}")
        history.append({"epoch": ep, "train_mse": float(tr_loss), "train_mae": float(tr_mae), "val_mse": float(va_loss), "val_mae": float(va_mae)})
    if best_state is not None:
        model.load_state_dict(best_state)
    te_loss, te_mae = _run(te_loader, train_flag=False, epoch=num_epochs + 1, phase="test")
    print(f"[TEST] MSE = {te_loss:.4f}")

    # save
    ckpt_path = workdir / "mv_dgt_ckpt.pt"
    torch.save({"state_dict": model.state_dict(), "meta": meta}, ckpt_path)
    print(f"[OK] wrote {ckpt_path}")

    if writer is not None:
        try:
            writer.flush()
            writer.close()
        except Exception:
            pass

    return {"val_mse": float(best), "test_mse": float(te_loss), "history": history}
