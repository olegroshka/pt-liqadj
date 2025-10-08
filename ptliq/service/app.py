from __future__ import annotations
from fastapi import FastAPI
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
        return ScoreResponse(preds_bps=[float(v) for v in y])

    return app
