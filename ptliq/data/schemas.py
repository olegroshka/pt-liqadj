from __future__ import annotations
from pydantic import BaseModel, Field
from datetime import date, datetime

class BondRef(BaseModel):
    isin: str
    issuer: str
    sector: str
    rating: str
    issue_date: date
    maturity: date
    coupon: float
    amount_out: float
    curve_bucket: str

class Trade(BaseModel):
    ts: datetime
    isin: str
    side: str = Field(regex="^(BUY|SELL)$")
    size: float
    price: float
    is_portfolio: bool = False
