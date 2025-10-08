from __future__ import annotations
from typing import List, Dict, Any
from pydantic import BaseModel, Field

class ScoreRequest(BaseModel):
    rows: List[Dict[str, Any]] = Field(default_factory=list)

class ScoreResponse(BaseModel):
    preds_bps: List[float]
