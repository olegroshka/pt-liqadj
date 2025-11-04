from __future__ import annotations
import json
import pandas as pd
from types import SimpleNamespace
from fastapi.testclient import TestClient

from ptliq.service.app import create_app
from ptliq.web.site import parse_rows, to_dataframe, filter_dataframe, call_api, build_ui


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
        {"isin": "US1", "f_a": 1, "f_b": 2},
        {"isin": "US2", "f_a": 3, "f_b": 4},
        {"isin": "US3", "f_a": 5, "f_b": 6},
    ]
    preds = [{"US1": 0.0}, {"US2": 1.0}, {"US3": 2.0}]

    df = to_dataframe(rows, preds)
    assert list(df.columns) == ["Portfolio Id", "Isin", "Liquidity score adjustment"]
    assert df.shape[0] == 3
    # order preserved and synthetic ids assigned
    assert df.loc[0, "Portfolio Id"] == "req 1" and df.loc[0, "Isin"] == "US1"
    assert df.loc[1, "Portfolio Id"] == "req 2" and df.loc[1, "Isin"] == "US2"
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
