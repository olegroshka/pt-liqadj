from __future__ import annotations
from fastapi import FastAPI
import numpy as np

from .dtypes import ScoreRequest, ScoreResponse
from .scoring import Scorer

def create_app(scorer: Scorer) -> FastAPI:
    app = FastAPI(title="pt-liqadj", version="0.1.0")

    @app.get("/health")
    def health():
        return {"ok": True, "in_dim": len(scorer.bundle.feature_names)}

    @app.post("/score", response_model=ScoreResponse)
    def score(req: ScoreRequest):
        y = scorer.score_many(req.rows)
        # extra guard before JSON serialization
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        preds = []
        for row, v in zip(req.rows, y):
            isin = row.get("isin")
            # ensure a string key and handle missing isin gracefully
            key = str(isin) if isin is not None else f"row_{len(preds)}"
            preds.append({key: float(v)})
        return ScoreResponse(preds_bps=preds)

    return app
