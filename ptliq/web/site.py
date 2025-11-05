from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path
import json
import pandas as pd
import logging

from .sample_payload import (
    load_samples_df,
    choose_portfolio_id,
    make_payload_for_portfolio,
    PayloadOptions,
    payload_to_compact_json,
)

ISIN = "Isin"

PORTFOLIO_ID = "Portfolio Id"

SIDE = "Side"

SCORE_ADJUSTMENT = "Liquidity Score Adjustment"

try:
    import gradio as gr  # type: ignore
except Exception:  # pragma: no cover
    gr = None  # lazy import in build_ui

from fastapi import FastAPI
from fastapi.testclient import TestClient


def parse_rows(json_text: str) -> List[Dict[str, Any]]:
    """
    Parse input JSON from the textarea.
    Accepts either an object with key "rows": [...], or a raw JSON list of objects.
    Returns list[dict]. Raises ValueError on malformed input.
    """
    txt = (json_text or "").strip()
    if not txt:
        return []
    try:
        obj = json.loads(txt)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")
    if isinstance(obj, dict) and "rows" in obj:
        rows = obj.get("rows")
    else:
        rows = obj
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        raise ValueError("Expected a JSON array of objects or an object with key 'rows'")
    return rows  # type: ignore[return-value]


def call_api(api: str | FastAPI, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Call the FastAPI /score endpoint with the provided rows.
    - If `api` is a string, treat it as base URL (e.g., http://127.0.0.1:8010).
    - If `api` is a FastAPI app instance, use TestClient to call in-process (useful for tests).
    Returns the `preds_bps` list of dicts.
    """
    payload = {"rows": rows}
    if isinstance(api, FastAPI):
        client = TestClient(api)
        r = client.post("/score", json=payload)
        r.raise_for_status()
        js = r.json()
    else:
        import requests  # lazy import
        base = api.rstrip("/")
        r = requests.post(f"{base}/score", json=payload, timeout=10)
        r.raise_for_status()
        js = r.json()
    preds = js.get("preds_bps", [])
    if not isinstance(preds, list):
        raise ValueError("API returned unexpected format: 'preds_bps' is not a list")
    return preds


def to_dataframe(rows: List[Dict[str, Any]], preds: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Combine input rows and predictions to a DataFrame with columns:
    - Portfolio Id (copied from request/response "portfolio_id")
    - Isin (copied from request/response "isin")
    - Side (copied from request/response "side")
    - Liquidity score adjustment (float value from response field "pred_bps")
    Order is preserved by index.
    """
    n = min(len(rows), len(preds))
    data: List[Tuple[str | None, str | None, str | None, float | None]] = []
    for i in range(n):
        row = rows[i]
        pred = preds[i] if i < len(preds) else {}
        # Prefer values from response; fall back to request row if missing
        portfolio_id = (pred or {}).get("portfolio_id", row.get("portfolio_id"))
        isin = (pred or {}).get("isin", row.get("isin"))
        side = (pred or {}).get("side", row.get("side"))
        score_value = (pred or {}).get("pred_bps", None)
        try:
            score = None if score_value is None else float(score_value)
        except Exception:
            score = None
        data.append(
            (
                None if portfolio_id is None else str(portfolio_id),
                None if isin is None else str(isin),
                None if side is None else str(side).lower(),
                score,
            )
        )
    df = pd.DataFrame(data, columns=[PORTFOLIO_ID, ISIN, SIDE, SCORE_ADJUSTMENT])
    return df


def filter_dataframe(df: pd.DataFrame, sel_portfolios: List[str] | None, sel_isins: List[str] | None, sel_side: List[str] | None = None) -> pd.DataFrame:
    """
    Filter DataFrame by selected Portfolio Ids, Isins, and Side.
    - Portfolio Id: multi-select via CheckboxGroup (selecting all behaves like no filter, but we still apply it)
    - Isin: multi-select via Dropdown; empty selection means no filter (ergonomic for hundreds of ISINs)
    - Side: multi-select via CheckboxGroup (two values: "buy", "sell"). If both selected or empty -> no filter.
    """
    out = df
    if sel_portfolios:
        out = out[out[PORTFOLIO_ID].isin(sel_portfolios)]
    if sel_isins:
        out = out[out[ISIN].isin(sel_isins)]
    if sel_side:
        out = out[out[SIDE].isin(sel_side)]
    return out.reset_index(drop=True)


def merge_append_override(prev_df: pd.DataFrame | None, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge DataFrames for Score accumulation:
    - Remove rows from prev_df whose Portfolio Id appears in new_df
    - Append new_df at the end
    - Preserve relative order of remaining previous rows and new rows
    """
    if prev_df is None or prev_df.empty:
        return new_df.copy()
    replace_ids = set([pid for pid in new_df[PORTFOLIO_ID].dropna().unique().tolist()])
    if len(replace_ids) == 0:
        return pd.concat([prev_df, new_df], ignore_index=True)
    keep_mask = ~prev_df[PORTFOLIO_ID].isin(list(replace_ids))
    merged = pd.concat([prev_df[keep_mask], new_df], ignore_index=True)
    return merged


def clear_dataframe(df: pd.DataFrame, sel_portfolios: List[str] | None, sel_isins: List[str] | None, sel_side: List[str] | None = None) -> pd.DataFrame:
    """
    Remove rows that are currently shown given the active filters.
    - If any filters are active, drop only matching rows.
    - If no filters are selected, drop all rows (return empty df with same columns).
    """
    if df is None or df.empty:
        return df
    # Determine rows to keep = original minus rows matching current filters
    if (sel_portfolios and len(sel_portfolios) > 0) or (sel_isins and len(sel_isins) > 0) or (sel_side and len(sel_side) > 0 and len(sel_side) < 2):
        shown = filter_dataframe(df, sel_portfolios, sel_isins, sel_side)
        keep = df.merge(shown, how="outer", indicator=True)
        keep = keep[keep["_merge"] == "left_only"].drop(columns=["_merge"]).reset_index(drop=True)
        return keep
    else:
        # No filters: clear all
        return pd.DataFrame(columns=df.columns)


def default_example_payload_json() -> str:
    """
    Return a compact JSON example payload with one JSON line per row and only the
    required fields: portfolio_id, isin, side, size. This is used as a fallback
    when no samples dataset is provided or if sample generation fails.
    """
    example_payload = {
        "rows": [
            {"portfolio_id": "01", "isin": "SIM0000000270", "side": "buy",  "size": 100000},
            {"portfolio_id": "01", "isin": "SIM0000000173", "side": "sell", "size": 50000},
            {"portfolio_id": "01", "isin": "SIM0000000467", "side": "buy",  "size": 250000},
        ]
    }
    return payload_to_compact_json(example_payload)


def build_ui(api: str | FastAPI, sample_source: Optional[str | Path] = None, workdir: Optional[str | Path] = None,
             n_rows: Optional[int] = None, size_jitter: float = 0.15, random_state: Optional[int] = 17):
    """
    Build a minimal Gradio UI around the scoring API.
    Allows JSON input, runs scoring, shows a grid, and supports multi-select filters.

    If `sample_source` (path to samples.parquet) or `workdir` (directory containing samples.parquet) is provided,
    the input textbox is pre-populated with a generated sample JSON derived from that dataset:
    - Pick a portfolio with available rows
    - Keep isin/side, jitter sizes slightly so it's not identical to training
    - Render each row on its own JSON line
    """
    global gr
    if gr is None:
        # delayed import if gradio was not present at import time
        import gradio as gr  # type: ignore

    logger = logging.getLogger("ptliq.web.site")
    logger.info(f"build_ui: sample_source={sample_source} workdir={workdir} n_rows={n_rows} size_jitter={size_jitter} random_state={random_state}")

    example_text: str
    try:
        if (sample_source is not None) or (workdir is not None):
            df = load_samples_df(source=sample_source, workdir=workdir)
            pid = choose_portfolio_id(df)
            opts = PayloadOptions(n_rows=n_rows, size_jitter=size_jitter, random_state=random_state)
            payload = make_payload_for_portfolio(df, pid, opts)
            example_text = payload_to_compact_json(payload)
            try:
                logger.info(f"build_ui: using samples-based payload from workdir={workdir or sample_source}; chosen portfolio_id={pid}; rows={len(payload.get('rows', []))}")
            except Exception:
                pass
        else:
            raise RuntimeError("no samples source provided")
    except Exception as e:
        logger.exception(f"build_ui: failed to prepare samples-based payload; falling back to default example. Reason: {e}")
        # Fallback static example (compact one-line-per-row, required fields only)
        example_text = default_example_payload_json()

    with gr.Blocks(title="Liquidity Adjustment Scoring") as demo:
        gr.Markdown("## Liquidity Scoring Demo\nPaste JSON payload and score. Output grid can be filtered.")
        with gr.Row():
            inp = gr.Textbox(
                label="JSON payload",
                value=example_text,
                lines=12,
                show_copy_button=True,
            )
        with gr.Row():
            btn = gr.Button("Score", variant="primary")
            btn_clear = gr.Button("Clear", variant="secondary")
        with gr.Row():
            portfolio_ms = gr.CheckboxGroup(choices=[], label="Filter: Portfolio Id", interactive=True)
            isin_ms = gr.Dropdown(choices=[], value=[], multiselect=True, label="Filter: Isin", filterable=True)
            side_ms = gr.CheckboxGroup(choices=["buy", "sell"], value=["buy", "sell"], label="Filter: Side", interactive=True)
        with gr.Row():
            grid = gr.Dataframe(
                value=pd.DataFrame(columns=[PORTFOLIO_ID, ISIN, SIDE, ("%s" % SCORE_ADJUSTMENT)]),
                headers=[PORTFOLIO_ID, ISIN, SIDE, SCORE_ADJUSTMENT],
                datatype=["str", "str", "str", "number"],
            )
            status = gr.Markdown(visible=False)

        # Keep server-side state simple to avoid frontend serialization issues
        state_df = gr.State(value=None)

        def _run(json_text: str, prev_df: pd.DataFrame | None):
            try:
                rows = parse_rows(json_text)
                preds = call_api(api, rows)
                new_df = to_dataframe(rows, preds)
                # Merge strategy: keep previous rows except those whose Portfolio Id appears in the new result, then append new rows.
                if prev_df is not None and not prev_df.empty:
                    # Identify portfolio ids to replace
                    replace_ids = set([pid for pid in new_df[PORTFOLIO_ID].dropna().unique().tolist()])
                    if len(replace_ids) > 0:
                        keep_mask = ~prev_df[PORTFOLIO_ID].isin(list(replace_ids))
                        merged = pd.concat([prev_df[keep_mask], new_df], ignore_index=True)
                    else:
                        merged = pd.concat([prev_df, new_df], ignore_index=True)
                else:
                    merged = new_df
                # populate filter options from merged df
                p_choices = sorted([x for x in merged[PORTFOLIO_ID].dropna().unique().tolist()])
                i_choices = sorted([x for x in merged[ISIN].dropna().unique().tolist()])
                s_choices = [x for x in ["buy", "sell"] if x in set(merged[SIDE].dropna().str.lower().unique().tolist())]
                if len(s_choices) == 0:
                    s_choices = ["buy", "sell"]
                # For large ISIN lists, default to no selection = show all; user can type to search and pick specific ones
                return (
                    merged,
                    gr.update(choices=p_choices, value=p_choices),
                    gr.update(choices=i_choices, value=[]),
                    gr.update(choices=s_choices, value=s_choices),
                    gr.update(visible=False, value=""),
                    merged,
                )
            except Exception as e:
                empty = pd.DataFrame(columns=[PORTFOLIO_ID, ISIN, SIDE, SCORE_ADJUSTMENT])
                return empty, gr.update(), gr.update(), gr.update(), gr.update(visible=True, value=f"Error: {e}"), empty

        def _apply_filters(sel_p: List[str], sel_i: List[str], sel_s: List[str], df: pd.DataFrame):
            if df is None or df.empty:
                return df
            return filter_dataframe(df, sel_p, sel_i, sel_s)


        def _clear(sel_p: List[str] | None, sel_i: List[str] | None, sel_s: List[str] | None, df: pd.DataFrame):
            # Remove currently shown rows; if no filters, clear all
            new_df = clear_dataframe(df, sel_p, sel_i, sel_s) if df is not None else df
            # Update filter options to reflect remaining rows
            if new_df is None or new_df.empty:
                return (
                    pd.DataFrame(columns=[PORTFOLIO_ID, ISIN, SIDE, SCORE_ADJUSTMENT]),
                    gr.update(choices=[], value=[]),
                    gr.update(choices=[], value=[]),
                    gr.update(choices=["buy", "sell"], value=["buy", "sell"]),
                    gr.update(visible=False, value=""),
                    pd.DataFrame(columns=[PORTFOLIO_ID, ISIN, SIDE, SCORE_ADJUSTMENT]),
                )
            p_choices = sorted(new_df[PORTFOLIO_ID].dropna().unique().tolist())
            i_choices = sorted(new_df[ISIN].dropna().unique().tolist())
            s_choices = [x for x in ["buy", "sell"] if x in set(new_df[SIDE].dropna().str.lower().unique().tolist())]
            if len(s_choices) == 0:
                s_choices = ["buy", "sell"]
            # With many ISINs, default to no selection (show all) after clear
            return (
                new_df,
                gr.update(choices=p_choices, value=p_choices),
                gr.update(choices=i_choices, value=[]),
                gr.update(choices=s_choices, value=s_choices),
                gr.update(visible=False, value=""),
                new_df,
            )

        btn.click(_run, inputs=[inp, state_df], outputs=[grid, portfolio_ms, isin_ms, side_ms, status, state_df])
        btn_clear.click(_clear, inputs=[portfolio_ms, isin_ms, side_ms, state_df], outputs=[grid, portfolio_ms, isin_ms, side_ms, status, state_df])
        portfolio_ms.change(_apply_filters, inputs=[portfolio_ms, isin_ms, side_ms, state_df], outputs=[grid])
        isin_ms.change(_apply_filters, inputs=[portfolio_ms, isin_ms, side_ms, state_df], outputs=[grid])
        side_ms.change(_apply_filters, inputs=[portfolio_ms, isin_ms, side_ms, state_df], outputs=[grid])

    return demo
