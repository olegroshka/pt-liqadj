from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Iterable
import json
import math
import random
import logging

import pandas as pd

REQUIRED_COLS = {"portfolio_id", "isin", "side", "size"}


@dataclass
class PayloadOptions:
    n_rows: Optional[int] = None  # limit number of rows for the portfolio
    size_jitter: float = 0.15     # +/- 15% uniform jitter
    random_state: Optional[int] = 17


def load_samples_df(source: Path | str | None = None, workdir: Path | str | None = None) -> pd.DataFrame:
    """
    Load samples.parquet. Priority:
    - If `source` is provided, read it.
    - Else if `workdir` is provided, read `workdir/samples.parquet`.
    Returns a DataFrame and validates the presence of required columns.
    Emits INFO/DEBUG logs describing the lookup path and validation.
    """
    logger = logging.getLogger("ptliq.web.sample_payload")
    path: Optional[Path] = None
    if source is not None:
        path = Path(source)
        logger.info(f"load_samples_df: using explicit source path: {path}")
    elif workdir is not None:
        path = Path(workdir) / "samples.parquet"
        logger.info(f"load_samples_df: using workdir path: {path}")
    else:
        logger.error("load_samples_df: neither source nor workdir provided")
        raise FileNotFoundError("samples.parquet path not provided")

    if not path.exists():
        logger.error(f"load_samples_df: samples.parquet not found at: {path}")
        raise FileNotFoundError(f"samples.parquet not found at: {path}")

    df = pd.read_parquet(path)
    logger.info(f"load_samples_df: loaded {path} with shape={df.shape} columns={list(df.columns)}")

    # Backward-compat: derive missing required columns from legacy schema
    try:
        if ("side" not in df.columns) and ("side_sign" in df.columns):
            def _map_side(v):
                try:
                    x = float(v)
                except Exception:
                    return None
                if x > 0:
                    return "buy"
                if x < 0:
                    return "sell"
                return None
            df["side"] = df["side_sign"].apply(_map_side)
            logger.info("load_samples_df: derived 'side' from 'side_sign' (>0=buy, <0=sell)")
        if ("size" not in df.columns) and ("log_size" in df.columns):
            def _map_size(v):
                try:
                    x = float(v)
                except Exception:
                    x = 0.0
                try:
                    s = math.exp(x)
                except Exception:
                    s = 0.0
                if not (s > 0.0):
                    s = 1.0
                try:
                    return int(max(1, round(s)))
                except Exception:
                    return 1
            df["size"] = df["log_size"].apply(_map_size)
            logger.info("load_samples_df: derived 'size' from 'log_size' via exp(log_size) → int; min=1")
    except Exception as e:
        logger.warning(f"load_samples_df: failed to derive legacy columns: {e}")

    # Validate required columns present
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        logger.error(f"load_samples_df: missing required columns: {sorted(missing)} at {path}")
        raise ValueError(f"samples.parquet missing required columns: {sorted(missing)}")
    try:
        n_ports = df['portfolio_id'].nunique()
        logger.info(f"load_samples_df: distinct portfolio_id count={n_ports}")
    except Exception:
        pass
    return df


def choose_portfolio_id(df: pd.DataFrame) -> str:
    """Pick a portfolio_id that has at least one row. Uses the first by appearance (stable)."""
    if df.empty:
        raise ValueError("empty samples DataFrame")
    # preserve original order
    for pid in df["portfolio_id"].tolist():
        if pd.notna(pid):
            return str(pid)
    # fallback to any unique
    vals = df["portfolio_id"].dropna().unique().tolist()
    if not vals:
        raise ValueError("no valid portfolio_id values in samples")
    return str(vals[0])


def _apply_size_jitter(sizes: Iterable[float | int], jitter: float, rnd: random.Random) -> List[int]:
    out: List[int] = []
    j = float(max(0.0, jitter))
    for s in sizes:
        try:
            base = float(s)
        except Exception:
            base = 0.0
        if base <= 0:
            base = 1.0
        lo = 1.0 - j
        hi = 1.0 + j
        factor = rnd.uniform(lo, hi)
        val = int(max(1, round(base * factor)))
        out.append(val)
    return out


def make_payload_for_portfolio(df: pd.DataFrame, portfolio_id: str,
                               options: PayloadOptions | None = None) -> Dict[str, Any]:
    """
    Build payload dict {"rows": [...]} for a single portfolio.
    - Keeps columns: portfolio_id, isin, side, size (size is jittered +/- size_jitter).
    - Returns Python dict, not a JSON string.
    """
    opts = options or PayloadOptions()
    pid = str(portfolio_id)
    sdf = df[df["portfolio_id"].astype(str) == pid].copy()
    if sdf.empty:
        raise ValueError(f"portfolio_id not found in samples: {portfolio_id}")
    if opts.n_rows is not None and opts.n_rows > 0:
        sdf = sdf.head(int(opts.n_rows))

    rnd = random.Random(opts.random_state)
    jittered = _apply_size_jitter(sdf["size"].tolist(), opts.size_jitter, rnd)

    rows: List[Dict[str, Any]] = []
    for (isin, side, size_val) in zip(sdf["isin"].tolist(), sdf["side"].tolist(), jittered):
        rows.append({
            "portfolio_id": pid,
            "isin": None if pd.isna(isin) else str(isin),
            "side": None if pd.isna(side) else str(side).lower(),
            "size": int(size_val),
        })
    return {"rows": rows}


def payload_to_compact_json(payload: Dict[str, Any]) -> str:
    """
    Serialize payload with one line per row. The outer structure is indented for readability,
    but each row dict is rendered compactly on a single line.
    """
    rows: List[Dict[str, Any]] = payload.get("rows", [])  # type: ignore
    head = "{\n  \"rows\": [\n"
    parts: List[str] = []
    for r in rows:
        parts.append("    " + json.dumps(r, separators=(",", ":")))
    body = ",\n".join(parts)
    tail = "\n  ]\n}"
    return head + body + tail


def prepare_samples_in_workdir(source: Path | str, workdir: Path | str) -> Path:
    """Copy samples parquet into model workdir as samples.parquet (overwrite)."""
    src = Path(source)
    dst_dir = Path(workdir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "samples.parquet"
    # Use pandas to preserve schema by re-writing
    df = pd.read_parquet(src)
    df.to_parquet(dst, index=False)
    return dst
