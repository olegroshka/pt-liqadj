import json
import numpy as np
import pandas as pd
import pytest
import torch
from pathlib import Path

from ptliq.training.mvdgt_loop import (
    MVDGTModelConfig, MVDGTTrainConfig, train_mvdgt
)

try:
    from tests.test_mvdgt_e2e import _generate_toy_world
    HAVE_GENERATOR = True
except Exception:
    HAVE_GENERATOR = False

@pytest.mark.skipif(not HAVE_GENERATOR, reason="toy-world generator not importable")
@pytest.mark.e2e
def test_diag_scalers_and_market_sign(tmp_path: Path, capfd):
    work, pyg_dir, out, ctx = _generate_toy_world(tmp_path / "toy_scaling", seed=61)
    cfg = MVDGTTrainConfig(
        workdir=work, pyg_dir=pyg_dir, outdir=out,
        epochs=5, lr=5e-3, weight_decay=1e-4, batch_size=128,
        seed=7, device="cpu", enable_tb=False, enable_tqdm=False,
        model=MVDGTModelConfig(hidden=32, heads=2, dropout=0.10, trade_dim=2, use_portfolio=True),
    )
    train_mvdgt(cfg)

    scaler = json.loads((out / "scaler.json").read_text())
    print(f"[DIAG] scaler.json = {scaler}")
    feat_names = json.loads((out / "feature_names.json").read_text())
    print(f"[DIAG] feature_names.json = {feat_names}")

    preproc_path = out / "market_preproc.json"
    if preproc_path.exists():
        pre = json.loads(preproc_path.read_text())
        print(f"[DIAG] market_preproc = {pre}")
        # crude sign check: correlation between (z-scored mkt) and y on TRAIN should have the same sign
        samp = pd.read_parquet(work / "samples.parquet")
        tr = samp[samp["split"] == "train"]
        mf = torch.load(ctx["files"]["market_context"] if "files" in ctx else (Path(work) / "market_context.pt"), map_location="cpu")["mkt_feat"] #todo: bad idea to use data/pyg but no time
        idx = torch.tensor(tr["date_idx"].astype(int).to_numpy())
        M = mf.index_select(0, idx)
        Mz = (M - torch.tensor(pre["mean"])) / torch.tensor([max(1e-6, float(s)) for s in pre["std"]])
        y = torch.tensor(tr["y"].astype(float).to_numpy())
        sign_emp = float(np.sign(np.corrcoef(Mz[:, 0].numpy(), y.numpy())[0, 1]))
        print(f"[DIAG] market_preproc.sign = {pre.get('sign', 1.0)} | empirical_sign(first_col) ~ {sign_emp}")
