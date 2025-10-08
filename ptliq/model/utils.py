# ptliq/model/utils.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


# ---------------------------
# Public container used by GNN
# ---------------------------
@dataclass
class GraphInputs:
    # per-trade (batch) fields
    node_ids: torch.Tensor                  # [B] ids of traded bonds
    cats: Dict[str, torch.Tensor]           # {'sector_code': [B], 'rating_code': [B]}
    nums: torch.Tensor                      # [B, n_num] numeric features
    port_nodes: torch.Tensor                # [B, T] node ids composing the portfolio context
    port_len: torch.Tensor                  # [B] valid lengths within T
    y: torch.Tensor | None                  # [B, 1] target (optional; may be injected later)

    # graph/global fields
    issuer_groups: Dict[int, torch.Tensor]  # issuer_id -> [Ni] member node ids
    sector_groups: Dict[int, torch.Tensor]  # sector_id -> [Ns] member node ids
    node_to_issuer: torch.Tensor            # [N] issuer id per node
    node_to_sector: torch.Tensor            # [N] sector id per node
    n_nodes: int                            # total universe size (distinct ISINs)


# ---------------------------
# Helpers
# ---------------------------
def _read_ranges(ranges_path: Path) -> Dict[str, Dict[str, str]]:
    with open(ranges_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _date_only(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts).dt.normalize()


def _encode_categorical(series: pd.Series) -> Tuple[pd.Series, Dict[str, int]]:
    vals = series.astype("category")
    codes = vals.cat.codes.astype(int)
    mapping = {str(cat): int(code) for code, cat in enumerate(vals.cat.categories)}
    return codes, mapping


def _curve_bucket_to_code(curve_bucket: pd.Series) -> pd.Series:
    # quick deterministic mapping (e.g., "5Y","10Y","30Y" -> 0/1/2)
    codes, _ = _encode_categorical(curve_bucket.fillna("UNK").astype(str))
    return codes


def _compute_days_to_mty(maturity: pd.Series, trade_date: pd.Series) -> pd.Series:
    md = pd.to_datetime(maturity).dt.normalize()
    td = pd.to_datetime(trade_date).dt.normalize()
    delta = (md - td).dt.days.clip(lower=0).fillna(0)
    return delta.astype(int)


def _build_universe(bonds: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    # assign stable node ids to each isin
    isins = bonds["isin"].astype(str).unique().tolist()
    isin_to_id = {isin: i for i, isin in enumerate(sorted(isins))}
    bonds["node_id"] = bonds["isin"].map(isin_to_id)
    return bonds, isin_to_id


def _build_groups(bonds: pd.DataFrame) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    # issuer/sector groups -> list of node ids
    issuer_groups: Dict[int, torch.Tensor] = {}
    sector_groups: Dict[int, torch.Tensor] = {}
    for issuer_code, df_g in bonds.groupby("issuer_code"):
        issuer_groups[int(issuer_code)] = torch.tensor(df_g["node_id"].values, dtype=torch.long)
    for sector_code, df_g in bonds.groupby("sector_code"):
        sector_groups[int(sector_code)] = torch.tensor(df_g["node_id"].values, dtype=torch.long)
    return issuer_groups, sector_groups


def _one_portfolio(node_id: int, issuer_id: int, sector_id: int,
                   issuer_groups: Dict[int, torch.Tensor],
                   sector_groups: Dict[int, torch.Tensor],
                   n_nodes: int,
                   max_port_items: int,
                   rng: np.random.Generator) -> Tuple[List[int], int]:
    """
    Build a light-weight portfolio context for a trade:
      - include the traded bond itself
      - include some names from same issuer/sector
      - fill remainder with random names from universe
    Return (list_of_node_ids (length T), valid_len)
    """
    items: List[int] = [node_id]

    # sample a few from same issuer/sector (if available)
    if issuer_id in issuer_groups:
        pool = issuer_groups[issuer_id].cpu().numpy().tolist()
        pool = [p for p in pool if p != node_id]
        rng.shuffle(pool)
        items.extend(pool[: min(4, len(pool))])

    if sector_id in sector_groups:
        pool = sector_groups[sector_id].cpu().numpy().tolist()
        pool = [p for p in pool if p != node_id and p not in items]
        rng.shuffle(pool)
        items.extend(pool[: min(8, len(pool))])

    # pad with random names
    while len(items) < max_port_items:
        items.append(rng.integers(0, n_nodes))

    # cap
    items = items[:max_port_items]
    valid_len = max(1, min(len(items), max_port_items))
    return items, valid_len


# ---------------------------
# Public API
# ---------------------------
def build_graph_inputs_for_split(rawdir: Path,
                                 ranges_path: Path,
                                 split: str,
                                 max_port_items: int = 64,
                                 seed: int = 1234) -> GraphInputs:
    """
    Construct GraphInputs directly from raw bonds/trades for the given split.
    No external feature pipeline is required.

    Features:
      - cats: sector_code, rating_code
      - nums: [amount_log, coupon, curve_code, days_to_mty]
      - portfolio: port_nodes [B,T], port_len [B]
    """
    rawdir = Path(rawdir)
    bonds = pd.read_parquet(rawdir / "bonds.parquet")
    trades = pd.read_parquet(rawdir / "trades.parquet")

    # encode issuer/sector/rating/curve
    bonds = bonds.copy()
    bonds["issuer_code"], _ = _encode_categorical(bonds["issuer"].astype(str))
    bonds["sector_code"], _ = _encode_categorical(bonds["sector"].astype(str))
    bonds["rating_code"], _ = _encode_categorical(bonds["rating"].astype(str))
    bonds["curve_code"] = _curve_bucket_to_code(bonds["curve_bucket"])

    bonds, isin_to_id = _build_universe(bonds)

    # date handling & split filter
    ranges = _read_ranges(Path(ranges_path))
    assert split in ranges, f"split '{split}' not in ranges.json"
    s = pd.to_datetime(ranges[split]["start"]).normalize()
    e = pd.to_datetime(ranges[split]["end"]).normalize()

    trades = trades.copy()
    trades["trade_date"] = _date_only(trades["ts"])
    trades = trades[(trades["trade_date"] >= s) & (trades["trade_date"] <= e)]

    # join bond meta
    df = trades.merge(bonds, on="isin", how="left", suffixes=("", "_b"))
    # safety drop na
    df = df.dropna(subset=["node_id", "sector_code", "rating_code", "issuer_code", "curve_code", "maturity", "amount_out", "coupon"])

    # cats
    sector_code = torch.tensor(df["sector_code"].astype(int).values, dtype=torch.long)
    rating_code = torch.tensor(df["rating_code"].astype(int).values, dtype=torch.long)

    # nums
    amount_log = np.log1p(df["amount_out"].astype(float).values)
    coupon = df["coupon"].astype(float).values
    curve_code = df["curve_code"].astype(int).values
    days_to_mty = _compute_days_to_mty(df["maturity"], df["trade_date"]).values

    nums = torch.tensor(
        np.stack([amount_log, coupon, curve_code.astype(float), days_to_mty.astype(float)], axis=1).astype(np.float32)
    )

    node_ids = torch.tensor(df["node_id"].astype(int).values, dtype=torch.long)

    # groups + mappings
    issuer_groups, sector_groups = _build_groups(bonds)
    node_to_issuer = torch.tensor(bonds.sort_values("node_id")["issuer_code"].astype(int).values, dtype=torch.long)
    node_to_sector = torch.tensor(bonds.sort_values("node_id")["sector_code"].astype(int).values, dtype=torch.long)
    n_nodes = int(bonds["node_id"].nunique())

    # portfolio construction
    rng = np.random.default_rng(seed)
    T = max_port_items
    port_nodes = torch.empty((len(df), T), dtype=torch.long)
    port_len = torch.empty((len(df),), dtype=torch.long)
    for i, (nid, iss, sec) in enumerate(zip(node_ids.tolist(),
                                            df["issuer_code"].astype(int).tolist(),
                                            df["sector_code"].astype(int).tolist())):
        items, vlen = _one_portfolio(nid, iss, sec, issuer_groups, sector_groups, n_nodes, T, rng)
        port_nodes[i] = torch.tensor(items, dtype=torch.long)
        port_len[i] = int(vlen)

    cats = {"sector_code": sector_code, "rating_code": rating_code}

    return GraphInputs(
        node_ids=node_ids,
        cats=cats,
        nums=nums,
        port_nodes=port_nodes,
        port_len=port_len,
        y=None,
        issuer_groups=issuer_groups,
        sector_groups=sector_groups,
        node_to_issuer=node_to_issuer,
        node_to_sector=node_to_sector,
        n_nodes=n_nodes,
    )


def _inject_portfolio_dependent_targets(gi: GraphInputs,
                                        w_issuer: float = 0.6,
                                        w_sector: float = 0.4,
                                        noise_std: float = 0.02) -> GraphInputs:
    """
    Create a synthetic target y that *depends on the portfolio composition*:
      y = w_issuer * frac(portfolio from same issuer) + w_sector * frac(portfolio from same sector) + noise
    This gives the GNN+Transformer a structural advantage over a baseline that ignores portfolios.
    """
    N = gi.node_ids.shape[0]
    T = gi.port_nodes.shape[1]

    # gather issuer/sector ids per target trade
    nid = gi.node_ids.long()
    tgt_issuer = gi.node_to_issuer[nid]          # [B]
    tgt_sector = gi.node_to_sector[nid]          # [B]

    # issuer/sector for all portfolio nodes
    pnodes = gi.port_nodes.long()                # [B, T]
    p_issuer = gi.node_to_issuer[pnodes]         # [B, T]
    p_sector = gi.node_to_sector[pnodes]         # [B, T]

    # valid mask
    arange_t = torch.arange(T, device=pnodes.device).view(1, T)
    valid = (arange_t < gi.port_len.view(-1, 1)).to(torch.float32)  # [B, T]

    # fractions
    same_iss = (p_issuer == tgt_issuer.view(-1, 1)).to(torch.float32) * valid
    same_sec = (p_sector == tgt_sector.view(-1, 1)).to(torch.float32) * valid
    denom = torch.clamp(valid.sum(dim=1), min=1.0)

    frac_iss = same_iss.sum(dim=1) / denom
    frac_sec = same_sec.sum(dim=1) / denom

    y = w_issuer * frac_iss + w_sector * frac_sec
    if noise_std > 0:
        y = y + noise_std * torch.randn_like(y)

    gi.y = y.view(-1, 1).to(torch.float32)
    return gi

def resolve_device(device: str | torch.device = "cpu") -> torch.device:
    """
    Resolve a device spec:
      - "auto" → cuda if available else cpu
      - "cpu"  → cpu
      - "cuda" or "cuda:{i}" → that CUDA device if available, else cpu
      - torch.device → returned as-is
    """
    if isinstance(device, torch.device):
        return device
    s = str(device).lower()
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if s.startswith("cuda"):
        return torch.device(s if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


# Public alias expected by the exp CLI
def inject_portfolio_dependent_targets(
    gi: GraphInputs,
    w_issuer: float = 0.6,
    w_sector: float = 0.4,
    noise_std: float = 0.02,
) -> GraphInputs:
    """Public wrapper around the private synthetic target injector."""
    return _inject_portfolio_dependent_targets(
        gi, w_issuer=w_issuer, w_sector=w_sector, noise_std=noise_std
    )