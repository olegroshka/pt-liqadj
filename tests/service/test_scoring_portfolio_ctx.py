from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import torch
import pytest

from ptliq.service.scoring import DGTScorer


class _CaptureModel(torch.nn.Module):
    """A tiny stub to capture inputs passed by DGTScorer to the model.
    It returns a zero tensor of shape [B].
    """
    def __init__(self):
        super().__init__()
        self.last_kwargs = None
        self.register_buffer("x_ref", None, persistent=False)

    def forward(self, x, *, anchor_idx, market_feat=None, pf_gid=None, port_ctx=None, trade_feat=None):  # noqa: D401
        # store a shallow snapshot (tensors kept by reference)
        self.last_kwargs = {
            "anchor_idx": anchor_idx,
            "market_feat": market_feat,
            "pf_gid": pf_gid,
            "port_ctx": port_ctx,
            "trade_feat": trade_feat,
        }
        b = int(anchor_idx.numel())
        return torch.zeros((b,), dtype=torch.float32, device=anchor_idx.device)


@pytest.mark.parametrize("model_dir", ["models/dgt0", "models/dgt_demo"], ids=["dgt0", "demo"])
def test_runtime_port_ctx_is_built_and_used(tmp_path: Path, model_dir: str):
    root = Path(__file__).resolve().parents[2]
    workdir = root / model_dir
    if not (workdir / "ckpt.pt").exists():
        pytest.skip(f"Artifacts not available at {workdir}")

    scorer = DGTScorer.from_dir(workdir)
    # monkeypatch the model with a capture stub
    cap = _CaptureModel().to(scorer.device)
    scorer.model = cap

    # Read graph nodes to get valid ISINs
    # Obtain ISINs from the scorer's internal mapping to avoid brittle meta dependencies
    isins = sorted([str(k) for k in scorer._isin_to_node.keys()])
    assert len(isins) >= 3, "Need at least 3 ISINs to build test baskets"

    def last_mkt_date() -> str | None:
        # Prefer scorer's precomputed market dates if available
        try:
            dates = getattr(scorer, "_mkt_dates", None)
            if dates is not None and len(dates) > 0:
                return str(pd.Timestamp(dates.max()).date())
        except Exception:
            pass
        return None

    asof = last_mkt_date()

    # Build two baskets sharing the same common line
    common = {"isin": isins[0], "side": "BUY", "size": 200_000}
    if asof:
        common["asof_date"] = asof

    P1 = [
        {"portfolio_id": "P1", **common},
        {"portfolio_id": "P1", "isin": isins[1], "side": "BUY", "size": 100_000},
        {"portfolio_id": "P1", "isin": isins[2], "side": "SELL", "size": 300_000},
    ]
    P2 = [
        {"portfolio_id": "P2", **common},
        {"portfolio_id": "P2", "isin": isins[3], "side": "SELL", "size": 500_000},
        {"portfolio_id": "P2", "isin": isins[4], "side": "SELL", "size": 200_000},
        {"portfolio_id": "P2", "isin": isins[5], "side": "BUY",  "size": 150_000},
    ]

    # Case A: runtime portfolio context must be constructed and passed
    _ = scorer.score_many(P1)
    kw1 = cap.last_kwargs
    assert kw1 is not None, "Model was not called"
    assert kw1["port_ctx"] is not None, "Expected runtime port_ctx when portfolio_id is present"
    assert kw1["pf_gid"] is not None, "Expected pf_gid vector when portfolio_id is present"
    pf_gid_1 = kw1["pf_gid"].detach().cpu().tolist()
    # All rows belong to group 0 in this single-portfolio request
    assert set(pf_gid_1) == {0}, f"pf_gid mapping incorrect: {pf_gid_1}"
    port_len_1 = kw1["port_ctx"]["port_len"].detach().cpu().tolist()
    assert port_len_1 == [len(P1)], f"port_len should equal number of items in P1, got {port_len_1}"

    # Case B: second basket, separate group indices expected
    _ = scorer.score_many(P2)
    kw2 = cap.last_kwargs
    port_len_2 = kw2["port_ctx"]["port_len"].detach().cpu().tolist()
    assert port_len_2 == [len(P2)], f"port_len should equal number of items in P2, got {port_len_2}"

    # Case C: if explicit pf_gid is supplied, runtime dynamic port_ctx should still be built
    P3 = [dict(r, pf_gid=7) for r in P1]  # explicit override (single group id)
    _ = scorer.score_many(P3)
    kw3 = cap.last_kwargs
    assert kw3["pf_gid"] is not None
    pf3 = kw3["pf_gid"].detach().cpu().tolist()
    assert set(pf3) == {7}, f"pf_gid should respect explicit values, got {pf3}"
    assert kw3["port_ctx"] is not None, "Expected dynamic port_ctx even when explicit pf_gid is provided"
    port_len_3 = kw3["port_ctx"]["port_len"].detach().cpu().tolist()
    assert port_len_3 == [len(P3)], f"port_len should equal number of items in P3, got {port_len_3}"


def test_portfolio_id_string_is_ignored_for_representation(tmp_path: Path):
    root = Path(__file__).resolve().parents[2]
    workdir = root / "models/dgt_demo"
    if not (workdir / "ckpt.pt").exists():
        pytest.skip("Artifacts not available for this test")
    scorer = DGTScorer.from_dir(workdir)
    cap = _CaptureModel().to(scorer.device)
    scorer.model = cap

    meta = json.loads((workdir / "mvdgt_meta.json").read_text())
    nodes_parq = meta["files"].get("graph_nodes")
    if not nodes_parq:
        # try fallback next to pyg_graph
        pyg_graph = meta["files"].get("pyg_graph")
        if pyg_graph:
            cand = Path(pyg_graph).parent / "graph_nodes.parquet"
            nodes_parq = str(cand) if cand.exists() else None
    if not nodes_parq:
        pytest.skip("graph_nodes parquet not found in meta and no fallback available")
    nodes_df = pd.read_parquet(nodes_parq)
    isins = nodes_df["isin"].astype(str).tolist()
    if len(isins) < 3:
        pytest.skip("Not enough nodes to build a basket")
    basket = [
        {"portfolio_id": "ABC", "isin": isins[0], "side": "BUY",  "size": 100_000},
        {"portfolio_id": "ABC", "isin": isins[1], "side": "SELL", "size": 200_000},
        {"portfolio_id": "ABC", "isin": isins[2], "side": "BUY",  "size": 300_000},
    ]
    _ = scorer.score_many(basket)
    kw_abc = cap.last_kwargs
    # Same rows but a different portfolio_id string should not change weights/lengths
    basket_xyz = [{**r, "portfolio_id": "XYZ"} for r in basket]
    _ = scorer.score_many(basket_xyz)
    kw_xyz = cap.last_kwargs

    for k in ("port_len", "port_w_abs_flat", "port_w_signed_flat"):
        t1 = kw_abc["port_ctx"][k].detach().cpu()
        t2 = kw_xyz["port_ctx"][k].detach().cpu()
        assert torch.allclose(t1, t2), f"{k} changed when only portfolio_id label changed"
