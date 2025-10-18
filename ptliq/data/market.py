from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import os
import pandas as pd


def _infer_date_range(years: int, start: Optional[str], end: Optional[str]) -> tuple[pd.Timestamp, pd.Timestamp]:
    tz_utc = timezone.utc
    if end:
        end_ts = pd.to_datetime(end).tz_localize(None)
    else:
        end_ts = pd.Timestamp(datetime.now(tz_utc)).tz_localize(None)
    if start:
        start_ts = pd.to_datetime(start).tz_localize(None)
    else:
        start_ts = end_ts - pd.Timedelta(days=365 * years + (years // 4))
    return pd.to_datetime(start_ts.date()), pd.to_datetime(end_ts.date())


def _load_fred_api_key(strict: bool = False) -> Optional[str]:
    """
    Load FRED_API_KEY from, in order:
      1) Environment variable FRED_API_KEY
      2) Project-local key file: configs/local/fred_api_key.txt (preferred)
      3) .env files in CWD or project root (legacy fallback)
    If strict is True and no key is found, raise a clear error telling the user how to create the local key file.
    """
    key = os.environ.get("FRED_API_KEY")
    if key:
        return key

    # Preferred local key file in project
    project_root = Path(__file__).resolve().parents[2]
    key_file = project_root / "configs" / "local" / "fred_api_key.txt"
    try:
        if key_file.exists():
            key = key_file.read_text(encoding="utf-8").strip()
            if key:
                os.environ["FRED_API_KEY"] = key
                return key
    except Exception:
        pass

    # Legacy fallback: .env in CWD or project root
    for p in [Path.cwd() / ".env", project_root / ".env"]:
        try:
            if p.exists():
                for line in p.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("FRED_API_KEY="):
                        key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if key:
                            os.environ["FRED_API_KEY"] = key
                            return key
        except Exception:
            continue

    if strict:
        raise RuntimeError(
            "FRED API key not found. Create a local file configs/local/fred_api_key.txt containing only your key, "
            "or set the FRED_API_KEY environment variable. Get a free key at: "
            "https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    return None


def fetch_fred_series(series_id: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Fetch a single FRED time series as a DataFrame with columns [date, value].

    This implementation requires a local API key and uses the official FRED API via fredapi.
    The key is read from configs/local/fred_api_key.txt (preferred) or the env var FRED_API_KEY.
    If not found, a clear error explains how to create the file.
    """
    # Require a key from local file or env
    api_key = _load_fred_api_key(strict=True)

    # Use official client
    try:
        from fredapi import Fred  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "fredapi is required for FRED access. Install with: pip install fredapi (or use .[market])"
        ) from e

    try:
        fred = Fred(api_key=api_key)
        s = fred.get_series(series_id, observation_start=start.date(), observation_end=end.date())
    except Exception as e:
        raise RuntimeError(f"Failed to fetch {series_id} from FRED using the API. {e}") from e

    if s is None or len(s) == 0:
        raise RuntimeError(f"No data received from FRED for {series_id} in the requested range.")

    s.index = pd.to_datetime(s.index)
    df = s.reset_index()
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def fetch_yahoo_single(ticker: str, start: pd.Timestamp, end: pd.Timestamp, interval: str = "1d") -> pd.DataFrame:
    """
    Fetch a single Yahoo Finance ticker and normalize to columns [date, close, volume] when available.
    """
    try:
        import yfinance as yf  # type: ignore
    except Exception:
        raise RuntimeError(
            "yfinance is required to download data from Yahoo Finance. Install with: pip install yfinance"
        )

    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    if df is None or df.empty:
        raise RuntimeError(f"No data received for {ticker} from Yahoo Finance in the requested range.")

    orig_index = df.index.copy()
    df = df.reset_index()
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

    # ensure 'date'
    if "date" not in df.columns:
        dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        if dt_cols:
            df = df.rename(columns={dt_cols[0]: "date"})
        elif "index" in df.columns:
            df["date"] = pd.to_datetime(df["index"], errors="coerce")
        else:
            df["date"] = pd.to_datetime(orig_index, errors="coerce")

    if "adj_close" in df.columns:
        df["close"] = df["adj_close"]

    candidates = ["date", "open", "high", "low", "close", "volume"]
    keep = [c for c in candidates if c in df.columns]
    if "date" not in keep:
        df["date"] = pd.to_datetime(orig_index, errors="coerce")
        keep = [c for c in candidates if c in df.columns]
    out = df[keep].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out = out.sort_values("date").reset_index(drop=True)
    return out
