from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch


@dataclass
class GraphInputs:
    # Per-sample
    node_ids: torch.Tensor           # [B]
    cats: Dict[str, torch.Tensor]    # name -> [B]
    nums: torch.Tensor               # [B, n_num]
    port_nodes: torch.Tensor         # [B, T] (-1 for pad)
    port_len: torch.Tensor           # [B]
    size_side_urg: torch.Tensor      # [B, 3]
    y: torch.Tensor                  # [B,1]

    # Global
    node_to_issuer: torch.Tensor     # [N]
    node_to_sector: torch.Tensor     # [N]
    issuer_groups: Dict[int, torch.Tensor]  # id -> LongTensor[node_ids]
    sector_groups: Dict[int, torch.Tensor]  # id -> LongTensor[node_ids]
    n_nodes: int


def _build_code_map(series: pd.Series) -> Tuple[pd.Series, int, Dict[str, int]]:
    u = sorted(series.unique())
    m = {v: i for i, v in enumerate(u)}
    return series.map(m).astype(int), len(u), m


def build_graph_inputs(
    bonds: pd.DataFrame,
    trades: pd.DataFrame,
    max_port_items: int = 128,
) -> GraphInputs:
    """
    Builds a single-day batch (or one batch per trade_date concatenated) using
    'portfolio = all trades on the same day'. This is a simple and deterministic
    way to create portfolio context for testing/eval.
    """
    # node ids
    isin2id = {isin: i for i, isin in enumerate(bonds["isin"].tolist())}
    trades = trades.copy()
    trades["node_id"] = trades["isin"].map(isin2id).astype(int)

    # codes
    bonds = bonds.copy()
    bonds["sector_code"], n_sector, _ = _build_code_map(bonds["sector"])
    bonds["rating_code"], n_rating, _ = _build_code_map(bonds["rating"])
    bonds["issuer_code"], n_issuer, _ = _build_code_map(bonds["issuer"])

    # join
    trades = trades.merge(
        bonds[["isin", "sector_code", "rating_code", "issuer_code", "coupon", "amount_out", "maturity"]],
        on="isin",
        how="left",
    )
    trades["trade_date"] = pd.to_datetime(trades["ts"]).dt.normalize()

    # group maps
    sector_groups = {int(k): torch.tensor([isin2id[s] for s in bonds[bonds["sector_code"] == k]["isin"].tolist()], dtype=torch.long)
                     for k in bonds["sector_code"].unique()}
    issuer_groups = {int(k): torch.tensor([isin2id[s] for s in bonds[bonds["issuer_code"] == k]["isin"].tolist()], dtype=torch.long)
                     for k in bonds["issuer_code"].unique()}
    node_to_sector = torch.tensor(bonds.sort_values("isin")["sector_code"].values, dtype=torch.long)
    node_to_issuer = torch.tensor(bonds.sort_values("isin")["issuer_code"].values, dtype=torch.long)

    # build per-sample tensors
    frames = []
    for d, g in trades.groupby("trade_date"):
        pset = torch.tensor(g["node_id"].values[:max_port_items], dtype=torch.long)
        T = int(pset.numel())
        B = len(g)

        frames.append({
            "node_ids": torch.tensor(g["node_id"].values, dtype=torch.long),
            "cats": {
                "sector_code": torch.tensor(g["sector_code"].values, dtype=torch.long),
                "rating_code": torch.tensor(g["rating_code"].values, dtype=torch.long),
            },
            "nums": torch.stack([
                torch.tensor(g["coupon"].values, dtype=torch.float32),
                torch.tensor(np.log1p(g["amount_out"].values), dtype=torch.float32),
                torch.tensor((g["side"] == "BUY").astype(int).values, dtype=torch.float32),
                torch.zeros(B, dtype=torch.float32),  # placeholder
            ], dim=1),
            "size_side_urg": torch.stack([
                torch.tensor(np.log1p(g["amount_out"].values), dtype=torch.float32),
                torch.tensor((g["side"] == "BUY").astype(int).values, dtype=torch.float32),
                torch.zeros(B, dtype=torch.float32),
            ], dim=1),
            "port_nodes": pset[None, :].repeat(B, 1),
            "port_len": torch.full((B,), T, dtype=torch.long),
            "y": torch.zeros(B, 1, dtype=torch.float32),  # if labels unknown yet
        })

    # concat by date
    node_ids = torch.cat([f["node_ids"] for f in frames], dim=0)
    cats = {
        "sector_code": torch.cat([f["cats"]["sector_code"] for f in frames], dim=0),
        "rating_code": torch.cat([f["cats"]["rating_code"] for f in frames], dim=0),
    }
    nums = torch.cat([f["nums"] for f in frames], dim=0)
    size_side_urg = torch.cat([f["size_side_urg"] for f in frames], dim=0)
    # Pad port_nodes to common width across dates for safe concatenation
    max_T = max(int(fr["port_nodes"].shape[1]) for fr in frames) if frames else 0
    padded_port_nodes = []
    for fr in frames:
        pn = fr["port_nodes"]
        T = int(pn.shape[1])
        if T < max_T:
            pad = torch.full((pn.shape[0], max_T - T), -1, dtype=pn.dtype)
            pn = torch.cat([pn, pad], dim=1)
        padded_port_nodes.append(pn)
    port_nodes = torch.cat(padded_port_nodes, dim=0) if frames else torch.empty((0, max_T), dtype=torch.long)
    port_len = torch.cat([f["port_len"] for f in frames], dim=0)
    y = torch.cat([f["y"] for f in frames], dim=0)

    return GraphInputs(
        node_ids=node_ids,
        cats=cats,
        nums=nums,
        port_nodes=port_nodes,
        port_len=port_len,
        size_side_urg=size_side_urg,
        y=y,
        node_to_issuer=node_to_issuer,
        node_to_sector=node_to_sector,
        issuer_groups=issuer_groups,
        sector_groups=sector_groups,
        n_nodes=len(bonds),
    )
