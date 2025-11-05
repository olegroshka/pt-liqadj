from __future__ import annotations
from fastapi import FastAPI, Request
import numpy as np
import logging
import time
import uuid

from .dtypes import ScoreRequest, ScoreResponse
from .scoring import Scorer


def create_app(scorer: Scorer) -> FastAPI:
    app = FastAPI(title="pt-liqadj", version="0.1.0")
    logger = logging.getLogger("ptliq.service.api")

    @app.middleware("http")
    async def _log_requests(request: Request, call_next):
        req_id = str(uuid.uuid4())
        start = time.perf_counter()
        path = request.url.path
        method = request.method
        request.state.req_id = req_id
        logger.debug(f"req_id={req_id} start {method} {path}")
        try:
            response = await call_next(request)
            status = getattr(response, "status_code", 0)
            dur_ms = (time.perf_counter() - start) * 1000.0
            logger.info(f"req_id={req_id} {method} {path} status={status} dur_ms={dur_ms:.2f}")
            return response
        except Exception as e:
            dur_ms = (time.perf_counter() - start) * 1000.0
            logger.exception(f"req_id={req_id} {method} {path} failed after {dur_ms:.2f} ms: {e}")
            raise

    @app.get("/health")
    def health(request: Request):
        rid = getattr(request.state, "req_id", "-")
        try:
            in_dim = len(scorer.bundle.feature_names)
        except Exception:
            in_dim = None
        logger.info(f"req_id={rid} /health in_dim={in_dim}")
        return {"ok": True, "in_dim": in_dim}

    @app.post("/score", response_model=ScoreResponse)
    def score(req: ScoreRequest, request: Request):
        rid = getattr(request.state, "req_id", "-")
        rows = req.rows or []
        n = len(rows)
        try:
            # Light request summary
            try:
                isins = [str(r.get("isin", "")).strip() for r in rows]
                uniq_isins = len(set([x for x in isins if x]))
                sides = [str(r.get("side", "")).strip().lower() for r in rows]
                n_buy = sum(1 for s in sides if s == "buy")
                n_sell = sum(1 for s in sides if s == "sell")
            except Exception:
                uniq_isins, n_buy, n_sell = None, None, None
            logger.info(f"req_id={rid} /score rows={n} uniq_isins={uniq_isins} side_dist={{buy:{n_buy},sell:{n_sell}}}")

            y = scorer.score_many(rows)
            # extra guard before JSON serialization
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            preds = []
            for row, v in zip(rows, y):
                isin = row.get("isin")
                portfolio_id = row.get("portfolio_id")
                side = row.get("side")
                item = {
                    "portfolio_id": None if portfolio_id is None else str(portfolio_id),
                    "isin": None if isin is None else str(isin),
                    "pred_bps": float(v),
                }
                if side is not None:
                    # Echo side back to response for UI filtering (expected values: "buy"/"sell")
                    try:
                        item["side"] = str(side)
                    except Exception:
                        item["side"] = None
                preds.append(item)
            try:
                # log quick stats of predictions
                vals = [float(p["pred_bps"]) for p in preds if p.get("pred_bps") is not None]
                if vals:
                    pmin, pmax = float(min(vals)), float(max(vals))
                else:
                    pmin = pmax = None
                logger.debug(f"req_id={rid} /score preds_count={len(preds)} min={pmin} max={pmax}")
            except Exception:
                pass
            return ScoreResponse(preds_bps=preds)
        except Exception as e:
            logger.exception(f"req_id={rid} /score failed: {e}")
            raise

    return app
