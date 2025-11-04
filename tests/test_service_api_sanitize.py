from fastapi.testclient import TestClient
from types import SimpleNamespace
from ptliq.service.app import create_app
import numpy as np

class DummyScorer:
    def __init__(self):
        self.bundle = SimpleNamespace(feature_names=["f_a", "f_b"])  # minimal for /health
    def score_many(self, rows):
        # return NaN/Inf values to check sanitization and order mapping
        return np.array([np.nan, np.inf, -np.inf, 1.5], dtype=float)

def test_score_sanitizes_and_preserves_order():
    scorer = DummyScorer()
    app = create_app(scorer)
    client = TestClient(app)

    rows = [
        {"isin": "US0001", "f_a": 1, "f_b": 2},
        {"isin": "US0002", "f_a": 3, "f_b": 4},
        {"isin": "US0003", "f_a": 5, "f_b": 6},
        {"isin": "US0004", "f_a": 7, "f_b": 8},
    ]
    r = client.post("/score", json={"rows": rows})

    assert r.status_code == 200
    js = r.json()
    preds = js["preds_bps"]
    isins = [d.get("isin") for d in preds]
    vals = [d.get("pred_bps") for d in preds]
    assert isins == [row["isin"] for row in rows]
    assert vals == [0.0, 0.0, 0.0, 1.5]
