from __future__ import annotations
import json
import pandas as pd
from types import SimpleNamespace
from fastapi.testclient import TestClient

from ptliq.service.app import create_app
from ptliq.web.site import parse_rows, to_dataframe, filter_dataframe, call_api, build_ui, merge_append_override

SCORE_ADJUSTMENT = "Liquidity Score Adjustment"


class DummyScorer:
    def __init__(self):
        self.bundle = SimpleNamespace(feature_names=["f_a", "f_b"])  # for /health
    def score_many(self, rows):
        # echo predictable sequence for easy assertions
        return [i * 1.0 for i in range(len(rows))]


def test_parse_rows_accepts_object_and_array():
    obj_payload = json.dumps({"rows": [{"isin": "US1", "f_a": 1, "f_b": 2}]})
    arr_payload = json.dumps([{"isin": "US1", "f_a": 1, "f_b": 2}])

    rows1 = parse_rows(obj_payload)
    rows2 = parse_rows(arr_payload)

    assert isinstance(rows1, list) and isinstance(rows1[0], dict)
    assert rows1 == rows2


def test_to_dataframe_and_filtering_roundtrip():
    rows = [
        {"portfolio_id": "P1", "isin": "US1", "f_a": 1, "f_b": 2},
        {"portfolio_id": "P1", "isin": "US2", "f_a": 3, "f_b": 4},
        {"portfolio_id": "P2", "isin": "US3", "f_a": 5, "f_b": 6},
    ]
    preds = [
        {"portfolio_id": "P1", "isin": "US1", "pred_bps": 0.0},
        {"portfolio_id": "P1", "isin": "US2", "pred_bps": 1.0},
        {"portfolio_id": "P2", "isin": "US3", "pred_bps": 2.0},
    ]

    df = to_dataframe(rows, preds)
    assert list(df.columns) == ["Portfolio Id", "Isin", "Side", ("%s" % SCORE_ADJUSTMENT)]
    assert df.shape[0] == 3
    # order preserved and ids copied from input/response
    assert df.loc[0, "Portfolio Id"] == "P1" and df.loc[0, "Isin"] == "US1"
    assert df.loc[1, "Portfolio Id"] == "P1" and df.loc[1, "Isin"] == "US2"
    # side is optional; with given inputs it's None
    assert df["Side"].isna().all()
    # filtering by isin
    df_f = filter_dataframe(df, sel_portfolios=None, sel_isins=["US2", "US3"])
    assert df_f["Isin"].tolist() == ["US2", "US3"]


def test_call_api_and_build_ui_smoke():
    scorer = DummyScorer()
    api = create_app(scorer)
    client = TestClient(api)
    # health works
    r = client.get("/health")
    assert r.status_code == 200

    # call_api with in-process app
    rows = [{"isin": "US1", "f_a": 1, "f_b": 2}, {"isin": "US2", "f_a": 3, "f_b": 4}]
    preds = call_api(api, rows)
    assert isinstance(preds, list) and len(preds) == 2
    # build_ui does not launch; just constructs Blocks
    ui = build_ui(api)
    assert ui is not None


def test_merge_append_override_accumulates_and_replaces():
    # initial score for P1 with two bonds
    df1 = pd.DataFrame(
        {
            "Portfolio Id": ["P1", "P1"],
            "Isin": ["US1", "US2"],
            SCORE_ADJUSTMENT: [0.1, 0.2],
        }
    )
    # next score for P2: should append
    df2 = pd.DataFrame(
        {
            "Portfolio Id": ["P2"],
            "Isin": ["US3"],
            SCORE_ADJUSTMENT: [0.3],
        }
    )
    merged = merge_append_override(df1, df2)
    assert merged.shape == (3, 3)
    assert merged["Portfolio Id"].tolist() == ["P1", "P1", "P2"]
    assert merged["Isin"].tolist() == ["US1", "US2", "US3"]

    # third score for P1 again: should replace old P1 rows, keep P2
    df3 = pd.DataFrame(
        {
            "Portfolio Id": ["P1"],
            "Isin": ["US4"],
            SCORE_ADJUSTMENT: [0.4],
        }
    )
    merged2 = merge_append_override(merged, df3)
    # P2 row kept, P1 rows replaced and appended after remaining rows
    assert merged2["Portfolio Id"].tolist() == ["P2", "P1"]
    assert merged2["Isin"].tolist() == ["US3", "US4"]
