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
        return ScoreResponse(preds_bps=[float(v) for v in y])

    return app
