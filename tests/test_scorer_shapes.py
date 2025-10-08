# tests/test_scorer_shapes.py
from pathlib import Path
import json, numpy as np
from ptliq.service.scoring import Scorer

def test_scorer_returns_1d(tmp_path: Path):
    # minimal fake bundle
    (tmp_path / "feature_names.json").write_text(json.dumps(["f_a","f_b"]))
    (tmp_path / "scaler.json").write_text(json.dumps({"mean":[0,0], "std":[1,1]}))
    (tmp_path / "train_config.json").write_text(json.dumps({"hidden":[1], "dropout":0.0}))
    # tiny 1x1 model checkpoint
    import torch
    from ptliq.model.baseline import MLPRegressor
    m = MLPRegressor(2, hidden=[1], dropout=0.0)
    torch.save(m.state_dict(), tmp_path / "ckpt.pt")
    s = Scorer.from_dir(tmp_path)
    y = s.score_many([{"f_a":0, "f_b":0}, {"f_a":1, "f_b":-1}])
    assert y.ndim == 1 and y.shape[0] == 2
