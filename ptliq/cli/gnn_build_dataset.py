from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import typer
import yaml

app = typer.Typer(no_args_is_help=True)

# -----------------------------
# small utilities
# -----------------------------
def _coalesce_col(df: pd.DataFrame, base: str) -> pd.Series:
    """Return a single canonical column from {base, base_x, base_y} (in that priority)."""
    if base in df.columns:
        return df[base]
    if f"{base}_x" in df.columns:
        return df[f"{base}_x"]
    if f"{base}_y" in df.columns:
        return df[f"{base}_y"]
    return pd.Series([np.nan] * len(df), index=df.index, dtype="object")


def _load_graph(graph_dir: Path):
    """
    Expect artifacts from `ptliq-gnn-build-graph`:
      issuer_groups.pt, sector_groups.pt, node_to_issuer.pt, node_to_sector.pt, (meta.json optional)
    """
    graph_dir = Path(graph_dir)
    ig = torch.load(graph_dir / "issuer_groups.pt", map_location="cpu")
    sg = torch.load(graph_dir / "sector_groups.pt", map_location="cpu")
    n2i = torch.load(graph_dir / "node_to_issuer.pt", map_location="cpu")
    n2s = torch.load(graph_dir / "node_to_sector.pt", map_location="cpu")

    meta = None
    mp = graph_dir / "meta.json"
    if mp.exists():
        try:
            meta = json.loads(mp.read_text())
        except Exception:
            meta = None
    return meta, n2i, n2s, ig, sg


def _try_load_index_lists(splits_dir: Path) -> Optional[Tuple[List[int], List[int], List[int]]]:
    """Try {train,val,test}.json with index lists."""
    d = Path(splits_dir)
    tj, vj, sj = d / "train.json", d / "val.json", d / "test.json"
    if tj.exists() and vj.exists() and sj.exists():
        return json.loads(tj.read_text()), json.loads(vj.read_text()), json.loads(sj.read_text())
    return None


def _try_load_ranges(splits_dir: Path, trades: pd.DataFrame) -> Optional[Tuple[List[int], List[int], List[int]]]:
    """
    Try ranges.json|yaml:
      {
        "train": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"},
        "val":   {"start": "...",         "end": "..."},
        "test":  {"start": "...",         "end": "..."}
      }
    """
    d = Path(splits_dir)
    ranges = None
    if (d / "ranges.json").exists():
        ranges = json.loads((d / "ranges.json").read_text())
    elif (d / "ranges.yaml").exists():
        ranges = yaml.safe_load((d / "ranges.yaml").read_text())
    if not ranges:
        return None

    date_col = "asof_date" if "asof_date" in trades.columns else "ts"
    tdates = pd.to_datetime(trades[date_col]).dt.date

    def _mask(span: Dict[str, str]):
        s = pd.to_datetime(span["start"]).date()
        e = pd.to_datetime(span["end"]).date()
        return ((tdates >= s) & (tdates <= e)).to_numpy().nonzero()[0].tolist()

    return _mask(ranges["train"]), _mask(ranges["val"]), _mask(ranges["test"])


def _fallback_split(trades: pd.DataFrame) -> Tuple[List[int], List[int], List[int]]:
    """60/20/20 chronological split if nothing provided."""
    order = trades.sort_values("ts").index.to_list()
    n = len(order)
    n_tr = max(1, int(0.6 * n))
    n_va = max(1, int(0.2 * n))
    idx_tr = order[:n_tr]
    idx_va = order[n_tr : n_tr + n_va]
    idx_te = order[n_tr + n_va :] or [order[-1]]
    return idx_tr, idx_va, idx_te


def _build_portfolio_fields(trades: pd.DataFrame, meta: Dict):
    """Return empty-but-valid portfolio fields for ctor compatibility."""
    n = len(trades)
    port_nodes_flat = torch.zeros((0,), dtype=torch.long)   # packed neighbors
    port_len = torch.zeros((n,), dtype=torch.long)          # per-row lengths
    port_legacy = torch.zeros((n, 0), dtype=torch.long)     # legacy shape (n x 0)
    return port_nodes_flat, port_len, port_legacy


def _mk_bundle(
    trades: pd.DataFrame,
    meta: Dict,
    issuer_groups: Dict[int, torch.Tensor],
    sector_groups: Dict[int, torch.Tensor],
    node_to_issuer: torch.Tensor,
    node_to_sector: torch.Tensor,
    rating_vocab: Optional[Dict[str, int]],
    target_col: str,
    derive_target: bool,
):
    """Build GraphInputs; try several ctor signatures for backward compatibility."""
    from ptliq.model.utils import GraphInputs  # late import

    isin2idx = meta["isin2idx"]
    node_ids = torch.as_tensor([int(isin2idx[i]) for i in trades["isin"].tolist()], dtype=torch.long)

    # categorical features
    sector2idx = meta.get("sector2idx", {})
    sector_code = torch.as_tensor([int(sector2idx[s]) for s in trades["sector"].tolist()], dtype=torch.long)

    rating_code = None
    if rating_vocab is not None and "rating" in trades.columns:
        rating_code = torch.as_tensor(
            [int(rating_vocab.get(str(r), 0)) for r in trades["rating"].tolist()],
            dtype=torch.long,
        )

    cats = {"sector_code": sector_code}
    if rating_code is not None:
        cats["rating_code"] = rating_code

    # numerical features
    size_np = trades["size"].astype(float).to_numpy(np.float32, copy=False)
    ref = trades["_ref_price"].astype(float).to_numpy()
    px = pd.to_numeric(trades.get("price", np.nan), errors="coerce").astype(float).to_numpy()
    ref_safe = np.where(ref == 0.0, 1.0, ref)
    delta_bps_np = 10_000.0 * (ref - px) / ref_safe
    delta_bps_np = np.nan_to_num(delta_bps_np, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    side_sign_np = (
        trades["side"].astype(str).str.upper().map({"SELL": 1.0, "BUY": -1.0}).fillna(0.0).to_numpy(np.float32)
    )
    signed_delta_bps_np = (side_sign_np * delta_bps_np).astype(np.float32)

    nums = {
        "size": torch.from_numpy(size_np),
        "delta_bps": torch.from_numpy(delta_bps_np),
        "side_sign": torch.from_numpy(side_sign_np),
        "signed_delta_bps": torch.from_numpy(signed_delta_bps_np),  # <-- NEW
    }

    # target
    if target_col in trades.columns:
        y_np = (
            pd.to_numeric(trades[target_col], errors="coerce")
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(np.float32)
        )
    else:
        y_np = signed_delta_bps_np if derive_target else np.zeros(len(trades), dtype=np.float32)
    y = torch.from_numpy(y_np)

    # portfolio auxiliaries
    port_nodes_flat, port_len, port_legacy = _build_portfolio_fields(trades, meta)

    kwargs = dict(
        n_nodes=int(meta["n_nodes"]),
        node_ids=node_ids,
        cats=cats,
        nums=nums,
        y=y,
        issuer_groups=issuer_groups,
        sector_groups=sector_groups,
        node_to_issuer=node_to_issuer,
        node_to_sector=node_to_sector,
    )

    # Try modern signature
    try:
        return GraphInputs(**kwargs, port_nodes=port_nodes_flat, port_len=port_len)
    except TypeError:
        pass
    # Legacy single "port"
    try:
        return GraphInputs(**kwargs, port=port_legacy)
    except TypeError:
        pass
    # Oldest: no portfolio args
    return GraphInputs(**kwargs)


# -----------------------------
# CLI
# -----------------------------
@app.command()
def build_dataset(
    trades: Path = typer.Option(..., help="trades.parquet (must include isin, size; target optional)"),
    bonds: Path = typer.Option(..., help="bonds.parquet (provides issuer/sector/rating to join)"),
    splits_dir: Path = typer.Option(..., help="either {train,val,test}.json or ranges.json/yaml"),
    graph_dir: Path = typer.Option(..., help="graph artifacts from gnn-build-graph"),
    outdir: Path = typer.Option(Path("data/gnn"), help="where to write train.pt/val.pt/test.pt"),
    target_col: str = typer.Option("y_bps", help="target column in trades; if missing, see --derive-target"),
    derive_target: bool = typer.Option(True, help="derive target if target_col missing"),
    ref_price_col: str = typer.Option("clean_price", help="bonds column with reference price"),
    default_par: float = typer.Option(100.0, help="fallback reference price"),
):
    """Join trades+bonds → build per-split GraphInputs and save to outdir."""
    outdir.mkdir(parents=True, exist_ok=True)

    # load
    tdf = pd.read_parquet(trades)
    bdf = pd.read_parquet(bonds)

    use_cols = [c for c in ["isin", "sector", "rating", "issuer", ref_price_col] if c in bdf.columns]
    bsmall = bdf[use_cols].copy()

    # join
    tdf = tdf.merge(bsmall, on="isin", how="left", suffixes=("_x", "_y"))
    tdf["sector"] = _coalesce_col(tdf, "sector")
    tdf["rating"] = _coalesce_col(tdf, "rating")
    tdf["issuer"] = _coalesce_col(tdf, "issuer")

    # ref price
    if ref_price_col in tdf.columns:
        tdf["_ref_price"] = pd.to_numeric(_coalesce_col(tdf, ref_price_col), errors="coerce").fillna(default_par)
    else:
        tdf["_ref_price"] = default_par

    # cleanup extra columns
    for c in [f"{ref_price_col}_x", f"{ref_price_col}_y", "sector_x", "sector_y", "rating_x", "rating_y", "issuer_x", "issuer_y"]:
        if c in tdf.columns:
            tdf.drop(columns=c, inplace=True)

    if tdf["sector"].isna().any():
        raise ValueError(
            f"After join, {int(tdf['sector'].isna().sum())} trade rows have missing sector. Ensure bonds cover all ISINs."
        )

    # derive target early so all splits have it
    if target_col not in tdf.columns and derive_target:
        side_sign = tdf["side"].astype(str).str.upper().map({"SELL": 1.0, "BUY": -1.0}).fillna(0.0).to_numpy()
        ref = tdf["_ref_price"].astype(float).to_numpy()
        px = pd.to_numeric(tdf.get("price", np.nan), errors="coerce").astype(float).to_numpy()
        ref_safe = np.where(ref == 0.0, 1.0, ref)
        delta_bps = 10_000.0 * (ref - px) / ref_safe
        signed_delta_bps = side_sign * np.nan_to_num(delta_bps, nan=0.0, posinf=0.0, neginf=0.0)
        tdf[target_col] = signed_delta_bps

    if target_col in tdf.columns:
        y = pd.to_numeric(tdf[target_col], errors="coerce").astype(float)
        y = y.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        tdf[target_col] = y

    # graph + meta
    meta, n2i, n2s, ig, sg = _load_graph(graph_dir)
    if meta is None:
        if "isin" not in bsmall.columns:
            raise ValueError("Cannot reconstruct meta: bonds are missing 'isin'.")
        n_nodes = int(n2i.numel()) if isinstance(n2i, torch.Tensor) else int(n2s.numel())
        isin_list = bsmall["isin"].astype(str).tolist()[:n_nodes]
        meta = {
            "n_nodes": n_nodes,
            "isin2idx": {isin: i for i, isin in enumerate(isin_list)},
            "sector2idx": {v: i for i, v in enumerate(sorted(bsmall["sector"].dropna().unique().tolist()))}
            if "sector" in bsmall.columns
            else {},
            "issuer2idx": {v: i for i, v in enumerate(sorted(bsmall["issuer"].dropna().unique().tolist()))}
            if "issuer" in bsmall.columns
            else {},
        }

    # splits
    splits = _try_load_index_lists(splits_dir) or _try_load_ranges(splits_dir, tdf) or _fallback_split(tdf)
    idx_tr, idx_va, idx_te = splits
    if any(len(x) == 0 for x in (idx_tr, idx_va, idx_te)):
        idx_tr, idx_va, idx_te = _fallback_split(tdf)

    # warn if constant target
    def _warn_constant(name, idx):
        if target_col in tdf.columns and len(idx) > 0:
            v = tdf.iloc[idx][target_col].to_numpy()
            if np.nanstd(v) == 0.0:
                typer.echo(f"[gnn-build-dataset] warning: {name} target is constant.", err=True)
    _warn_constant("train", idx_tr)
    _warn_constant("val", idx_va)
    _warn_constant("test", idx_te)

    # rating vocab
    rvocab = None
    if "rating" in bsmall.columns:
        rvocab = {v: i for i, v in enumerate(sorted(bsmall["rating"].dropna().unique().tolist()))}

    # bundles
    gi_tr = _mk_bundle(tdf.iloc[idx_tr].reset_index(drop=True), meta, ig, sg, n2i, n2s, rvocab, target_col, derive_target)
    gi_va = _mk_bundle(tdf.iloc[idx_va].reset_index(drop=True), meta, ig, sg, n2i, n2s, rvocab, target_col, derive_target)
    gi_te = _mk_bundle(tdf.iloc[idx_te].reset_index(drop=True), meta, ig, sg, n2i, n2s, rvocab, target_col, derive_target)

    # save
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(gi_tr, outdir / "train.pt")
    torch.save(gi_va, outdir / "val.pt")
    torch.save(gi_te, outdir / "test.pt")

    typer.echo("Wrote bundles to " + str(outdir))
