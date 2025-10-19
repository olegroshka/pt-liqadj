from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import typer
from rich import print

from ptliq.features.build import build_features

# -----------------------
# CLI
# -----------------------
app = typer.Typer(no_args_is_help=True)

# Allow invoking without subcommand by forwarding to 'features'
@app.callback(invoke_without_command=True)
def _default(
    ctx: typer.Context,
    rawdir: Path = typer.Option(Path("data/raw/sim"), help="Raw parquet dir"),
    splits: Path = typer.Option(None, help="Path to ranges.json from ptliq-split"),
    outdir: Path = typer.Option(Path("data/features"), help="Output base dir"),
    run_id: str = typer.Option("exp001", help="Run identifier"),
):
    if ctx.invoked_subcommand is None:
        if splits is None:
            raise typer.BadParameter("--splits is required when invoking without a subcommand")
        build_minimal_features(rawdir=rawdir, splits=splits, outdir=outdir, run_id=run_id)

# -----------------------
# Existing minimal features command (preserved)
# -----------------------
@app.command("features")
def build_minimal_features(
    rawdir: Path = typer.Option(Path("data/raw/sim"), help="Raw parquet dir"),
    splits: Path = typer.Option(..., help="Path to ranges.json from ptliq-split"),
    outdir: Path = typer.Option(Path("data/features"), help="Output base dir"),
    run_id: str = typer.Option("exp001", help="Run identifier"),
):
    """
    Build minimal model-ready features and write train/val/test parquet files.
    """
    frames = build_features(rawdir, splits)
    out = Path(outdir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    for k, df in frames.items():
        path = out / f"{k}.parquet"
        df.to_parquet(path, index=False)
        print(f"  • wrote {k}: {path} (rows={len(df)})")
    print(f"[bold green]FEATURES READY[/bold green] → {out}")

# -----------------------
# Helpers & guards for graph featurization
# -----------------------

def _require_cols(df: pd.DataFrame, cols: List[str], where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise typer.BadParameter(f"[{where}] required columns missing: {missing}")

def _tb_from_tenor(t: float) -> str:
    t = float(t)
    if t < 3: return "0-3y"
    if t < 5: return "3-5y"
    if t < 7: return "5-7y"
    if t < 10: return "7-10y"
    return "10-15y"

def _extract_pf_base(x: str) -> Optional[str]:
    if not isinstance(x, str):
        return None
    m = re.match(r"^(PF_\d{8})", x)
    return m.group(1) if m else None

def _factorize(series: pd.Series) -> Tuple[pd.Series, Dict[str, int]]:
    s = series.astype(str).fillna("__NA__")
    cats = sorted(s.unique().tolist())
    idx = {v: i for i, v in enumerate(cats)}
    return s.map(idx), idx

def _one_hot(s: pd.Series) -> Tuple[np.ndarray, List[str]]:
    s = s.fillna("__NA__").astype(str)
    classes = sorted(s.unique().tolist())
    mapper = {c: i for i, c in enumerate(classes)}
    X = np.zeros((len(s), len(classes)), dtype=np.float32)
    X[np.arange(len(s)), s.map(mapper).values] = 1.0
    return X, classes

def _print_relation_counts(tag: str, edges_df: pd.DataFrame) -> None:
    vc = (edges_df["relation"].value_counts() if "relation" in edges_df.columns else pd.Series(dtype=int))
    typer.echo(f"\n[{tag}] relation counts:")
    if len(vc) == 0:
        typer.echo("  (none)")
    else:
        for r, c in vc.sort_index().items():
            typer.echo(f"  {r:<14} {int(c)}")

# --- Market helpers: robust loaders and calendar alignment ---

def _load_series(path: Path, date_candidates=("date","asof_date","trade_date","ts"),
                 value_candidates=("value","close","oas","oas_bps","MOVE","VIX")) -> pd.DataFrame:
    """
    Returns DataFrame[asof_date (date), value (float)].
    Fail-fast if no usable cols or empty after cleaning.
    """
    df = pd.read_parquet(path)
    dcol = next((c for c in date_candidates if c in df.columns), None)
    vcol = next((c for c in value_candidates if c in df.columns), None)
    if not dcol or not vcol:
        raise typer.BadParameter(f"[market] expected a date col in {date_candidates} and value col in {value_candidates} for {path}")
    out = df[[dcol, vcol]].rename(columns={dcol: "asof_date", vcol: "value"})
    out["asof_date"] = pd.to_datetime(out["asof_date"], errors="coerce").dt.normalize().dt.date
    out = out.dropna(subset=["asof_date","value"]).copy()
    if out.empty:
        raise typer.BadParameter(f"[market] empty after cleaning: {path}")
    return out


def _align_calendar(trades: pd.DataFrame, sers: dict[str,pd.DataFrame], ffill_limit: int = 3) -> pd.DataFrame:
    """
    Build a business-day calendar covering all trade_dt in trades, align and limited forward-fill.
    Returns a single DataFrame indexed by asof_date with columns named after the keys in `sers`.
    """
    cal = pd.to_datetime(trades["trade_dt"]).dt.normalize().dt.date
    idx = pd.Index(sorted(pd.Series(cal).dropna().unique()), name="asof_date")
    out = pd.DataFrame(index=idx)
    for name, df in sers.items():
        x = (df.set_index("asof_date").sort_index()
               .reindex(idx).astype(float)
               .ffill(limit=ffill_limit))
        if x.isna().any().any():
            missing = int(x["value"].isna().sum()) if "value" in x.columns else int(x.isna().sum().sum())
            raise typer.BadParameter(f"[market] missing {missing} values for {name} after limited ffill; increase limit or fix source.")
        out[name] = x["value"].values if "value" in x.columns else x.iloc[:,0].values
    out = out.reset_index()
    return out


def _with_derived(df: pd.DataFrame, col: str, win: int = 20) -> pd.DataFrame:
    s = pd.Series(df[col].astype(float).values)
    df[f"{col}_chg_1d"] = s.diff(1)
    df[f"{col}_chg_5d"] = s.diff(5)
    roll = s.rolling(win, min_periods=max(5, win//4))
    mu = roll.mean(); sd = roll.std().replace(0.0, np.nan)
    df[f"{col}_z_{win}d"] = (s - mu) / (sd + 1e-8)
    return df

# --- NEW: portfolio line-item weights (per portfolio_id, trade_dt) ---
def _compute_portfolio_weights(trades: pd.DataFrame, node_id_map: dict) -> pd.DataFrame:
    t = trades.copy()
    _require_cols(t, ["isin", "dv01_dollar", "side", "trade_dt", "portfolio_id"], "trades/portfolio-weights")
    # normalize date + clean ids
    t["trade_dt"] = pd.to_datetime(t["trade_dt"], errors="coerce").dt.normalize()
    t["portfolio_id"] = t["portfolio_id"].replace(["", "None", "nan", "NaN"], np.nan)
    t = t.dropna(subset=["portfolio_id", "trade_dt", "isin", "dv01_dollar", "side"]).copy()

    # sign convention == COTRADE (BUY/CBUY=+1, SELL/CSELL=-1) for consistency
    t["sign"] = (
        t["side"].astype(str).str.upper()
         .map({"CBUY": 1.0, "BUY": 1.0, "CSELL": -1.0, "SELL": -1.0})
         .fillna(0.0)
    )

    # per-basket sums and ranks
    key = ["portfolio_id", "trade_dt"]
    t["dv01_abs"] = t["dv01_dollar"].astype(float).abs()
    denom = t.groupby(key)["dv01_abs"].transform("sum").replace(0.0, np.nan)
    t["w_dv01_abs_frac"] = (t["dv01_abs"] / denom).fillna(0.0).clip(0.0, 1.0)
    # ranks (1 = largest |DV01|) — compute within group safely
    t["rank_dv01_desc"] = t.groupby(key)["dv01_abs"].rank(ascending=False, method="min")
    n_in_group = t.groupby(key)["isin"].transform("size")
    t["rank_pct"] = ((t["rank_dv01_desc"] - 1) / (n_in_group - 1)).fillna(0.0)

    # signed fraction using same sign convention as COTRADE
    t["w_dv01_signed_frac"] = t["w_dv01_abs_frac"] * t["sign"]

    # keep only baskets with at least 2 lines (eligible portfolios)
    t = t[n_in_group >= 2].copy()

    # attach node ids (drop orphan ISINs if any)
    t["node_id"] = t["isin"].map(node_id_map)
    t = t.dropna(subset=["node_id"]).copy()
    t["node_id"] = t["node_id"].astype(int)

    # stable group id for downstream (int)
    t = t.sort_values(key + ["node_id"]).reset_index(drop=True)
    gid_map = {k: i for i, k in enumerate(sorted(t[key].drop_duplicates().itertuples(index=False, name=None)))}
    t["pf_gid"] = t[key].apply(tuple, axis=1).map(gid_map).astype(np.int32)

    keep_cols = [
        "pf_gid", "portfolio_id", "trade_dt",
        "isin", "node_id",
        "w_dv01_abs_frac", "w_dv01_signed_frac",
        "rank_dv01_desc", "rank_pct",
        "dv01_abs"
    ]
    return t[keep_cols]

# -----------------------
# Pair constructors
# -----------------------

def _make_pairs_from_groups(nodes: pd.DataFrame, key_col: str, rel: str, w: float, different_issuer_only: bool = False):
    if key_col not in nodes.columns:
        return []
    out = []
    for _, grp in nodes.groupby(key_col):
        isins = grp["isin"].tolist()
        issuers = grp.get("issuer", pd.Series([None] * len(grp))).tolist()
        for i in range(len(isins) - 1):
            for j in range(i + 1, len(isins)):
                if different_issuer_only and issuers[i] == issuers[j]:
                    continue
                out.append((isins[i], isins[j], rel, float(w)))
    return out

def _rating_near_pairs(nodes: pd.DataFrame):
    if "rating_num" not in nodes.columns:
        return []
    out = []
    for _, grp in nodes.groupby("rating_num"):
        isins = grp["isin"].tolist()
        for i in range(len(isins) - 1):
            for j in range(i + 1, len(isins)):
                out.append((isins[i], isins[j], "RATING_NEAR", 0.30))
    rvals = sorted(nodes["rating_num"].dropna().unique().tolist())
    rset = set(rvals)
    for r in rvals:
        if r + 1 in rset:
            left = nodes.loc[nodes["rating_num"] == r, "isin"].tolist()
            right = nodes.loc[nodes["rating_num"] == r + 1, "isin"].tolist()
            for i in left:
                for j in right:
                    out.append((i, j, "RATING_NEAR", 0.15))
    return out

# -----------------------
# Co-trade pairs (fail-fast; no fallback)
# -----------------------

def _bond_similarity(bi: pd.Series, bj: pd.Series) -> float:
    sim = 0.0
    if "sector" in bi and "sector" in bj and pd.notna(bi["sector"]) and pd.notna(bj["sector"]) and bi["sector"] == bj["sector"]:
        sim += 0.5
    if "rating_num" in bi and "rating_num" in bj and pd.notna(bi["rating_num"]) and pd.notna(bj["rating_num"]):
        gap = abs(int(bi["rating_num"]) - int(bj["rating_num"]))
        sim += max(0.0, 0.3 - 0.1 * gap)
    if "tenor_years" in bi and "tenor_years" in bj and pd.notna(bi["tenor_years"]) and pd.notna(bj["tenor_years"]):
        tdiff = abs(float(bi["tenor_years"]) - float(bj["tenor_years"]))
        sim += float(np.exp(-tdiff / 5.0)) * 0.2
    return float(np.clip(sim, 0.0, 1.0))


def _cotrade_pairs_failfast(
    trades: pd.DataFrame,
    bonds: pd.DataFrame,
    expo_scale: float = 5e4,
    target_min: int = 200,
    target_max: int = 300,
    random_state: int = 17,
):
    rng = np.random.default_rng(random_state)
    t = trades.copy()
    _require_cols(t, ["isin", "side", "dv01_dollar"], "trades")
    if "trade_dt" in t.columns:
        t["trade_dt"] = pd.to_datetime(t["trade_dt"], errors="coerce").dt.normalize()
    elif "exec_time" in t.columns:
        t["trade_dt"] = pd.to_datetime(t["exec_time"], errors="coerce").dt.normalize()
    else:
        raise typer.BadParameter("[trades] require trade_dt or exec_time")

    max_dt = t["trade_dt"].dropna().max()
    t["decay"] = np.exp(-((max_dt - t["trade_dt"]).dt.days.fillna(0)) / 30.0)

    t["sign"] = (
        t["side"].astype(str).str.upper().map({"CBUY": 1, "BUY": 1, "CSELL": -1, "SELL": -1}).astype(float)
    )

    t = t.dropna(subset=["isin", "sign", "dv01_dollar", "trade_dt"]).copy()
    if t.empty:
        raise typer.BadParameter("[trades] empty after cleaning required columns.")

    if "portfolio_id" not in t.columns:
        raise typer.BadParameter("[trades] require portfolio_id column to build co-trade edges.")

    groups: List[pd.DataFrame] = []

    t["portfolio_id"] = t["portfolio_id"].replace(["", "None", "nan", "NaN"], np.nan)

    # (1) exact (portfolio_id, trade_dt)
    g = t[t["portfolio_id"].notna()].groupby(["portfolio_id", "trade_dt"])
    for _, gg in g:
        if len(gg) >= 2:
            groups.append(gg)

    # (2) coalesce PF_YYYYMMDD_xxx → PF_YYYYMMDD and chunk
    base = t[t["portfolio_id"].notna()].copy()
    base["pf_base"] = base["portfolio_id"].map(_extract_pf_base)
    base = base[base["pf_base"].notna()]
    for (_, dt), gg in base.groupby(["pf_base", "trade_dt"]):
        if len(gg) >= 2:
            idx = gg.index.to_numpy()
            rng.shuffle(idx)
            pos = 0
            while pos < len(idx):
                batch_size = int(rng.integers(target_min, target_max + 1))
                batch_idx = idx[pos : pos + batch_size]
                if len(batch_idx) >= 2:
                    groups.append(gg.loc[batch_idx])
                pos += batch_size

    if len(groups) == 0:
        raise typer.BadParameter("[co-trade] found no eligible (portfolio_id, trade_dt) groups (>=2).")

    bidx = bonds.set_index("isin")
    acc: Dict[Tuple[str, str], float] = defaultdict(float)

    for gg in groups:
        rows = gg[["isin", "sign", "dv01_dollar", "decay"]].to_dict("records")
        n = len(rows)
        for a in range(n - 1):
            ia = rows[a]
            for b in range(a + 1, n):
                ib = rows[b]
                i, j = ia["isin"], ib["isin"]
                if i == j:
                    continue
                u, v = (i, j) if i <= j else (j, i)
                s = float(ia["sign"]) * float(ib["sign"])
                expo = min(abs(float(ia["dv01_dollar"])), abs(float(ib["dv01_dollar"])))/ float(expo_scale)
                expo = float(np.clip(expo, 0.0, 5.0))
                decay = float(min(ia["decay"], ib["decay"]))
                try:
                    bi, bj = bidx.loc[i], bidx.loc[j]
                except Exception:
                    sim = 0.0
                else:
                    sim = _bond_similarity(bi, bj)
                acc[(u, v)] += decay * s * expo * float(sim)

    out = [(i, j, "COTRADE", float(w)) for (i, j), w in acc.items() if i != j and abs(w) > 1e-12]
    if not out:
        raise typer.BadParameter("[co-trade] produced zero weighted pairs — check dv01_dollar/side/portfolio_id.")
    return out

# -----------------------
# Graph build (nodes, edges, sparsify)
# -----------------------

@app.command("graph")
def featurize_graph(
    bonds: Path = typer.Option(..., help="bonds.parquet"),
    trades: Path = typer.Option(..., help="trades.parquet"),
    outdir: Path = typer.Option(Path("data/graph"), help="Output directory"),
    expo_scale: float = typer.Option(5e4, help="Exposure scaling (dv01_dollar/expo_scale)"),
    target_min: int = typer.Option(200, help="Min chunk when coalescing PF_YYYYMMDD base"),
    target_max: int = typer.Option(300, help="Max chunk when coalescing PF_YYYYMMDD base"),
    cotrade_q: float = typer.Option(0.85, help="Quantile for abs(weight) cut on COTRADE"),
    cotrade_topk: int = typer.Option(20, help="Per-node TopK after quantile cut"),
    issuer_topk: int = typer.Option(20),
    sector_topk: int = typer.Option(8),
    rating_topk: int = typer.Option(4),
    curve_topk: int = typer.Option(4),
    currency_topk: int = typer.Option(0),
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    bdf = pd.read_parquet(bonds)
    tdf = pd.read_parquet(trades)

    _require_cols(bdf, ["isin", "issuer", "sector", "rating", "tenor_years"], "bonds")
    nodes = bdf.drop_duplicates("isin").copy().reset_index(drop=True)
    nodes["issuer_name"] = nodes["issuer"].astype(str)
    nodes["issuer_id"], _ = _factorize(nodes["issuer"].astype(str))
    nodes["sector_id"], _ = _factorize(nodes["sector"].astype(str))
    nodes["node_id"] = pd.factorize(nodes["isin"].astype(str))[0]
    rating_map = {"AAA": 1, "AA": 2, "A": 3, "BBB": 4, "BB": 5, "B": 6}
    nodes["rating_num"] = nodes["rating"].map(rating_map)
    if nodes["rating_num"].isna().any():
        raise typer.BadParameter("[bonds] unexpected rating symbols; expected one of AAA,AA,A,BBB,BB,B")

    if "curve_bucket" not in nodes.columns:
        nodes["curve_bucket"] = nodes["tenor_years"].astype(float).map(_tb_from_tenor)
    if "currency" not in nodes.columns:
        nodes["currency"] = np.nan

    node_id_map = dict(zip(nodes["isin"], nodes["node_id"]))

    # --- NEW: compute and save portfolio line-item weights and group index ---
    try:
        port_lines = _compute_portfolio_weights(tdf, node_id_map)
    except Exception as e:
        raise
    if port_lines.empty:
        raise typer.BadParameter("[portfolio-weights] found no eligible (portfolio_id, trade_dt) groups (>=2).")

    # Integrity checks: unit-sum per group, bounds, no NaNs
    sum_per_g = port_lines.groupby("pf_gid")["w_dv01_abs_frac"].sum()
    if not np.allclose(sum_per_g.to_numpy(), 1.0, atol=1e-6):
        bad = sum_per_g[(sum_per_g - 1.0).abs() > 1e-6]
        raise typer.BadParameter(f"[portfolio-weights] unit-sum check failed for groups: {bad.index.tolist()} values={bad.values.tolist()}")
    if port_lines[["w_dv01_abs_frac"]].isna().any().any() or (port_lines["w_dv01_abs_frac"] < 0).any() or (port_lines["w_dv01_abs_frac"] > 1).any():
        raise typer.BadParameter("[portfolio-weights] invalid weights: NaNs or out-of-bounds [0,1].")

    port_groups = (
        port_lines.groupby(["pf_gid","portfolio_id","trade_dt"], as_index=False)
        .agg(n_lines=("node_id","size"), sum_dv01_abs=("dv01_abs","sum"))
        .assign(sum_w_abs=lambda df: 1.0)
    )

    p_lines_path  = outdir / "portfolio_lines.parquet"
    p_groups_path = outdir / "portfolio_groups.parquet"
    port_lines.to_parquet(p_lines_path, index=False)
    port_groups.to_parquet(p_groups_path, index=False)
    typer.echo(f"Saved portfolio context:\n - {p_lines_path}\n - {p_groups_path}")

    # --- NEW: market features (MOVE / IG-OAS / HY-OAS / VIX) ---
    try:
        market_dir = Path("data/raw")
        move_p = market_dir / "move.parquet"
        ig_p   = market_dir / "ig_oas.parquet"
        hy_p   = market_dir / "hy_oas.parquet"
        vix_p  = market_dir / "vix.parquet"
        for p in [move_p, ig_p, hy_p, vix_p]:
            if not p.exists():
                raise typer.BadParameter(f"[market] missing file: {p}")

        move = _load_series(move_p, value_candidates=("value","MOVE","close"))
        ig   = _load_series(ig_p,   value_candidates=("oas_bps","value","close"))
        hy   = _load_series(hy_p,   value_candidates=("oas_bps","value","close"))
        vix  = _load_series(vix_p,  value_candidates=("value","VIX","close"))

        aligned = _align_calendar(tdf, {
            "MOVE_lvl": move,
            "IG_OAS_bps": ig,
            "HY_OAS_bps": hy,
            "VIX_lvl": vix,
        })

        aligned = _with_derived(aligned, "MOVE_lvl", win=20)
        aligned = _with_derived(aligned, "VIX_lvl",  win=20)
        aligned["HYIG_spread_bps"]  = aligned["HY_OAS_bps"] - aligned["IG_OAS_bps"]
        aligned["IG_OAS_chg_5d"]    = aligned["IG_OAS_bps"].astype(float).diff(5)
        aligned["HY_OAS_chg_5d"]    = aligned["HY_OAS_bps"].astype(float).diff(5)
        aligned["HYIG_chg_5d"]      = aligned["HYIG_spread_bps"].astype(float).diff(5)

        aligned = aligned.fillna(0.0)

        # sanity checks
        if aligned.isna().any().any():
            raise typer.BadParameter("[market] NaNs remain in market features after fills/derivatives.")

        mkt_path = outdir / "market_features.parquet"
        meta_path = outdir / "market_meta.json"
        aligned.to_parquet(mkt_path, index=False)

        mkt_meta = {
            "asof_key": "asof_date",
            "raw_files": ["move.parquet","ig_oas.parquet","hy_oas.parquet","vix.parquet"],
            "feature_names": [c for c in aligned.columns if c != "asof_date"],
            "ffill_limit_days": 3,
            "rolling_windows": {"MOVE_z_20d": 20, "VIX_z_20d": 20},
        }
        meta_path.write_text(json.dumps(mkt_meta, indent=2))
        typer.echo(f"Saved market features: {mkt_path}")
    except Exception as e:
        raise

    issuer_edges = _make_pairs_from_groups(nodes, "issuer", "ISSUER_SIBLING", 1.00)
    sector_edges = _make_pairs_from_groups(nodes, "sector_id", "SECTOR", 0.25, different_issuer_only=True)
    rating_edges = _rating_near_pairs(nodes)
    curve_edges = _make_pairs_from_groups(nodes, "curve_bucket", "CURVE_BUCKET", 0.20, different_issuer_only=True)
    currency_edges = (
        _make_pairs_from_groups(nodes, "currency", "CURRENCY", 0.10, different_issuer_only=True)
        if "currency" in nodes.columns and nodes["currency"].notna().any()
        else []
    )

    cotrade_edges = _cotrade_pairs_failfast(
        trades=tdf, bonds=bdf, expo_scale=expo_scale, target_min=target_min, target_max=target_max
    )

    edges_all = issuer_edges + sector_edges + rating_edges + curve_edges + currency_edges + cotrade_edges
    edges = pd.DataFrame(edges_all, columns=["src_isin", "dst_isin", "relation", "weight"])
    edges = edges[edges["src_isin"] != edges["dst_isin"]].copy()
    edges["src_id"] = edges["src_isin"].map(node_id_map)
    edges["dst_id"] = edges["dst_isin"].map(node_id_map)
    edges = edges.dropna(subset=["src_id", "dst_id"]).copy()
    edges["src_id"] = edges["src_id"].astype(int)
    edges["dst_id"] = edges["dst_id"].astype(int)
    u = edges[["src_id", "dst_id"]].min(axis=1)
    v = edges[["src_id", "dst_id"]].max(axis=1)
    edges["src_id"] = u
    edges["dst_id"] = v
    edges = edges.groupby(["src_id", "dst_id", "relation"], as_index=False)["weight"].sum()

    def _prune_topk(df: pd.DataFrame, rel: str, k: int) -> pd.DataFrame:
        if rel not in df["relation"].unique():
            return df
        if k <= 0:
            return df[df["relation"] != rel]
        sub = df[df["relation"] == rel].copy()
        keep_idx = []
        for col in ["src_id", "dst_id"]:
            srt = sub.sort_values([col, "weight"], ascending=[True, False])
            keep_idx.append(srt.groupby(col, as_index=False).head(k).index)
        mask = df.index.isin(np.concatenate(keep_idx))
        return pd.concat([df[df["relation"] != rel], df.loc[mask & (df["relation"] == rel)]], ignore_index=True).drop_duplicates()

    for rel, k in [
        ("ISSUER_SIBLING", issuer_topk),
        ("SECTOR", sector_topk),
        ("RATING_NEAR", rating_topk),
        ("CURVE_BUCKET", curve_topk),
        ("CURRENCY", currency_topk),
    ]:
        edges = _prune_topk(edges, rel, k)

    if "COTRADE" not in edges["relation"].unique():
        raise typer.BadParameter("[co-trade] no COTRADE edges after clique pruning.")
    cot = edges[edges["relation"] == "COTRADE"].copy()
    non = edges[edges["relation"] != "COTRADE"].copy()
    if cot.empty:
        raise typer.BadParameter("[co-trade] empty cotrade slab before sparsification.")
    cot["absw"] = cot["weight"].abs()
    tau = float(cot["absw"].quantile(float(cotrade_q)))
    keep = cot[cot["absw"] >= tau]
    min_frac = 0.15
    if len(keep) < min_frac * len(cot):
        tau = float(cot["absw"].quantile(1.0 - min_frac))
        keep = cot[cot["absw"] >= tau]
    keep = keep.copy()
    keep.loc[:, "relation"] = np.where(keep["weight"] >= 0, "COTRADE_CO", "COTRADE_X")
    keep.loc[:, "weight"] = keep["absw"]
    keep = keep.drop(columns=["absw"])

    idx = []
    for col in ["src_id", "dst_id"]:
        srt = keep.sort_values([col, "weight"], ascending=[True, False])
        idx.append(srt.groupby(col, as_index=False).head(int(cotrade_topk)).index)
    keep = keep.loc[np.concatenate(idx)].drop_duplicates()

    edges = pd.concat([non, keep], ignore_index=True)
    edges = edges.groupby(["src_id", "dst_id", "relation"], as_index=False)["weight"].sum()

    _print_relation_counts("AFTER PRUNE", edges)
    typer.echo("COTRADE_CO edges: " + str(int((edges["relation"] == "COTRADE_CO").sum())))
    typer.echo("COTRADE_X edges: " + str(int((edges["relation"] == "COTRADE_X").sum())))

    nodes_out = nodes[[
        "node_id",
        "isin",
        "issuer",
        "issuer_name",
        "sector",
        "sector_id",
        "rating",
        "rating_num",
        "curve_bucket",
        "tenor_years",
        "currency",
    ]].copy()

    edges_out = edges.copy()
    tot = (
        edges_out.groupby(["src_id", "dst_id"], as_index=False)["weight"].sum().rename(columns={"weight": "total_weight"})
    )
    edges_out = edges_out.merge(tot, on=["src_id", "dst_id"], how="left")
    rel2id = {r: i for i, r in enumerate(sorted(edges_out["relation"].unique().tolist()))}
    edges_out["relation_id"] = edges_out["relation"].map(rel2id)

    nodes_path = outdir / "graph_nodes.parquet"
    edges_path = outdir / "graph_edges.parquet"
    npz_path = outdir / "edge_index.npz"
    rel_path = outdir / "rel2id.json"

    # Save nodes/edges as Parquet for speed and schema fidelity
    nodes_out.to_parquet(nodes_path, index=False)
    edges_out.to_parquet(edges_path, index=False)

    # Also persist a compact edge bundle for quick loading in PyG
    edge_index = np.vstack([edges_out["src_id"].values, edges_out["dst_id"].values]).astype(np.int64)
    edge_weight = edges_out["weight"].values.astype(np.float32)
    relation_id = edges_out["relation_id"].values.astype(np.int64)
    np.savez_compressed(npz_path, edge_index=edge_index, edge_weight=edge_weight, relation_id=relation_id)
    rel_path.write_text(json.dumps(rel2id, indent=2))

    typer.echo(f"\nSaved:\n - {nodes_path}\n - {edges_path}\n - {npz_path}\n - {rel_path}")

    try:
        from ptliq.validate.graph import validate_graph_artifacts
        validate_graph_artifacts(nodes_path, edges_path, npz_path, outdir / "reports")
    except Exception as e:
        typer.echo(f"[validate] graph validation failed: {e}", err=True)

# -----------------------
# PyG assembly (features + symmetric edges)
# -----------------------

@app.command("pyg")
def featurize_pyg(
    graph_dir: Path = typer.Option(Path("data/graph"), help="Dir with graph_nodes.csv, graph_edges.csv, edge_index.npz"),
    outdir: Path = typer.Option(Path("data/pyg"), help="Output dir for pyg_graph.pt and feature_meta.json"),
) -> None:
    # Prefer Parquet; fall back to CSV for backward compatibility
    nodes_parq = graph_dir / "graph_nodes.parquet"
    edges_parq = graph_dir / "graph_edges.parquet"
    nodes_csv = graph_dir / "graph_nodes.csv"
    edges_csv = graph_dir / "graph_edges.csv"
    npz_path = graph_dir / "edge_index.npz"

    if nodes_parq.exists():
        nodes = pd.read_parquet(nodes_parq)
    elif nodes_csv.exists():
        nodes = pd.read_csv(nodes_csv)
    else:
        raise typer.BadParameter(f"[pyg] nodes file not found in {graph_dir} (expected graph_nodes.parquet or graph_nodes.csv)")

    if edges_parq.exists():
        edges = pd.read_parquet(edges_parq)
    elif edges_csv.exists():
        edges = pd.read_csv(edges_csv)
    else:
        raise typer.BadParameter(f"[pyg] edges file not found in {graph_dir} (expected graph_edges.parquet or graph_edges.csv)")

    npz = np.load(npz_path)

    N = len(nodes)
    if N == 0:
        raise typer.BadParameter("[pyg] nodes table is empty.")
    outdir.mkdir(parents=True, exist_ok=True)

    def _z(x: np.ndarray) -> np.ndarray:
        x = x.astype(float)
        mu = np.nanmean(x)
        sd = np.nanstd(x) + 1e-8
        return ((x - mu) / sd).reshape(-1, 1).astype(np.float32)

    x_rating = _z(nodes["rating_num"].to_numpy()) if "rating_num" in nodes.columns else np.zeros((N, 1), np.float32)
    x_tenor = _z(nodes["tenor_years"].to_numpy()) if "tenor_years" in nodes.columns else np.zeros((N, 1), np.float32)

    oh_sector, sector_classes = _one_hot(nodes["sector_id"].astype(str) if "sector_id" in nodes else pd.Series(["0"] * N))
    oh_curve, curve_classes = _one_hot(nodes["curve_bucket"] if "curve_bucket" in nodes else pd.Series(["UNK"] * N))
    oh_ccy, ccy_classes = _one_hot(
        nodes["currency"] if "currency" in nodes and nodes["currency"].notna().any() else pd.Series(["__NA__"] * N)
    )

    x_np = np.hstack([x_rating, x_tenor, oh_sector, oh_curve, oh_ccy]).astype(np.float32)

    issuer_index = nodes["issuer"].fillna("__NA__").astype(str)
    issuer_idx_map = {v: i for i, v in enumerate(sorted(issuer_index.unique().tolist()))}
    issuer_id = issuer_index.map(issuer_idx_map).astype(int).to_numpy()

    src = edges["src_id"].astype(int).values
    dst = edges["dst_id"].astype(int).values
    et = edges["relation_id"].astype(int).values
    ew = edges["weight"].astype(float).values

    edge_index = np.vstack([np.concatenate([src, dst]), np.concatenate([dst, src])]).astype(np.int64)
    edge_type = np.concatenate([et, et]).astype(np.int64)
    edge_weight = np.concatenate([ew, ew]).astype(np.float32)

    try:
        from torch_geometric.data import Data
    except Exception as ie:
        raise typer.BadParameter("[pyg] torch-geometric is required to build the dataset. Install with: pip install torch-geometric") from ie

    data = Data(
        x=torch.from_numpy(x_np),
        edge_index=torch.from_numpy(edge_index),
        edge_type=torch.from_numpy(edge_type),
        edge_weight=torch.from_numpy(edge_weight),
        num_nodes=int(N),
    )
    data.issuer_index = torch.from_numpy(issuer_id)

    torch.save(data, outdir / "pyg_graph.pt")
    meta = {
        "node_feature_schema": {
            "columns": (
                ["z_rating_num", "z_tenor_years"]
                + [f"onehot_sector_{c}" for c in sector_classes]
                + [f"onehot_curve_{c}" for c in curve_classes]
                + [f"onehot_ccy_{c}" for c in ccy_classes]
            ),
            "issuer_index": True,
        },
        "relations": sorted(edges["relation"].unique().tolist()),
        "feature_dims": {
            "x_dim": int(x_np.shape[1]),
            "num_relations": int(len(set(edges["relation_id"].tolist()))),
            "num_nodes": int(N),
            "num_edges_directed": int(edge_index.shape[1]),
        },
    }
    # --- NEW: optional portfolio context packing (ragged tensors) ---
    p_lines_path = graph_dir / "portfolio_lines.parquet"
    if p_lines_path.exists():
        pl = pd.read_parquet(p_lines_path)
        if {"pf_gid","node_id","w_dv01_abs_frac","w_dv01_signed_frac"}.issubset(pl.columns) and len(pl) > 0:
            pl = pl.sort_values(["pf_gid","rank_dv01_desc","node_id"]).reset_index(drop=True)
            grp = pl.groupby("pf_gid", sort=True, as_index=False)
            lengths = grp.size()["size"].astype(np.int64).to_numpy()
            offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])

            port_nodes_flat  = torch.as_tensor(pl["node_id"].to_numpy(np.int64, copy=False))
            port_w_abs_flat  = torch.as_tensor(pl["w_dv01_abs_frac"].to_numpy(np.float32, copy=False))
            port_w_sgn_flat  = torch.as_tensor(pl["w_dv01_signed_frac"].to_numpy(np.float32, copy=False))
            port_len         = torch.as_tensor(lengths, dtype=torch.long)
            port_offsets     = torch.as_tensor(offsets, dtype=torch.long)

            port_ctx = {
                "port_nodes_flat": port_nodes_flat.contiguous(),
                "port_w_abs_flat": port_w_abs_flat.contiguous(),
                "port_w_signed_flat": port_w_sgn_flat.contiguous(),
                "port_len": port_len.contiguous(),
                "port_offsets": port_offsets.contiguous(),
            }
            torch.save(port_ctx, outdir / "portfolio_context.pt")

            p_index = (pl[["pf_gid","portfolio_id","trade_dt"]].drop_duplicates()
                       .sort_values("pf_gid").reset_index(drop=True))
            p_index.to_parquet(outdir / "portfolio_index.parquet", index=False)

            meta["portfolio_context"] = {
                "num_groups": int(len(port_len)),
                "total_lines": int(int(port_len.sum())),
                "tensors": ["port_nodes_flat","port_w_abs_flat","port_w_signed_flat","port_len","port_offsets"],
                "index_file": "portfolio_index.parquet",
                "weight_fields": ["w_dv01_abs_frac","w_dv01_signed_frac"],
                "signed_sign_convention": "BUY=+1, SELL=-1 (matches COTRADE)",
            }
        else:
            typer.echo("[pyg] portfolio_lines.parquet present but missing required columns; skipping.", err=True)
    else:
        typer.echo("[pyg] no portfolio_lines.parquet found; skipping portfolio context packing.")

    # --- NEW: pack market context (date-indexed) ---
    mkt_path = graph_dir / "market_features.parquet"
    if mkt_path.exists():
        mkt = pd.read_parquet(mkt_path).copy()
        if "asof_date" not in mkt.columns:
            raise typer.BadParameter("[pyg] market_features.parquet missing 'asof_date'")
        mkt["asof_date"] = pd.to_datetime(mkt["asof_date"], errors="coerce").dt.normalize().dt.date
        mkt = mkt.dropna(subset=["asof_date"]).sort_values("asof_date").reset_index(drop=True)

        feat_cols = [c for c in mkt.columns if c != "asof_date"]
        mkt_dates = torch.as_tensor(pd.to_datetime(mkt["asof_date"]).astype("int64").to_numpy(), dtype=torch.long)
        mkt_feat  = torch.as_tensor(mkt[feat_cols].to_numpy(np.float32, copy=False))

        market_ctx = {"mkt_dates": mkt_dates.contiguous(), "mkt_feat": mkt_feat.contiguous()}
        torch.save(market_ctx, outdir / "market_context.pt")

        idx = (mkt.assign(row_idx=lambda df: np.arange(len(df), dtype=np.int64))[ ["asof_date","row_idx"] ])
        idx.to_parquet(outdir / "market_index.parquet", index=False)

        meta["market_context"] = {
            "num_days": int(len(mkt_dates)),
            "num_features": int(mkt_feat.size(1)),
            "feature_names": feat_cols,
            "index_file": "market_index.parquet",
        }
        (outdir / "feature_meta.json").write_text(json.dumps(meta, indent=2))
        typer.echo(f"Saved market context: {outdir/'market_context.pt'}")
    else:
        typer.echo("[pyg] market_features.parquet not found; skipping market context.")

    (outdir / "feature_meta.json").write_text(json.dumps(meta, indent=2))
    typer.echo(f"Saved:\n - {outdir/'pyg_graph.pt'}\n - {outdir/'feature_meta.json'}")

    try:
        from ptliq.validate.dataset import validate_pyg
        validate_pyg(outdir / "pyg_graph.pt", outdir / "reports")
    except Exception as e:
        typer.echo(f"[validate] dataset validation failed: {e}", err=True)

# expose Typer app
app = app

if __name__ == "__main__":
    app()
