from pydantic import BaseModel
from datetime import date
from enum import Enum


class Signal(str, Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    PASS = "pass"


class TechnicalSnapshot(BaseModel):
    ticker: str
    date: date
    price: float
    rsi_14: float | None = None
    macd_signal: str | None = None  # "bullish" / "bearish" / "neutral"
    above_sma_50: bool | None = None
    above_sma_200: bool | None = None
    volume_vs_avg: float | None = None  # ratio: today's vol / 20d avg
    atr_14: float | None = None


class FundamentalData(BaseModel):
    ticker: str
    revenue_current: float | None = None
    revenue_prior: float | None = None
    revenue_growth_pct: float | None = None
    seeking_alpha_rating: str | None = None


class InstitutionalData(BaseModel):
    ticker: str
    institutional_buyers: int | None = None
    institutional_sellers: int | None = None
    net_institutional: int | None = None
    short_interest_pct: float | None = None
    short_interest_change: float | None = None


class SeekingAlphaData(BaseModel):
    ticker: str
    value: int = 0           # 1-12 factor grade
    growth: int = 0
    momentum: int = 0
    profitability: int = 0
    eps_revisions: int = 0
    analyst_count: int = 0
    mean_score: float = 0.0  # 1-5 Wall Street consensus
    rating: str = "N/A"


class StockScore(BaseModel):
    ticker: str
    date: date
    technical_score: float = 0.0
    fundamental_score: float = 0.0
    institutional_score: float = 0.0
    sa_score: float = 0.0
    composite_score: float = 0.0
    signal: Signal = Signal.PASS
    entry_price: float | None = None
    stop_loss: float | None = None
    target_price: float | None = None
    notes: str = ""
