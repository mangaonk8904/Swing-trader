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


# ── Chart / Multi-Timeframe Setup Models ──────────────────────────────────────


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"
    NONE = "none"


class SetupStatus(str, Enum):
    ACTIVE = "at_entry"        # price is inside the entry zone right now
    APPROACHING = "approaching"  # within ~1 ATR of the zone
    WAIT = "wait"              # valid setup, price far from entry
    NO_SETUP = "no_setup"      # no acceptable setup found


class Zone(BaseModel):
    """A price zone (order block, fair-value gap, moving-average band)."""
    kind: str                  # "bullish_ob" / "bearish_ob" / "bullish_fvg" / "bearish_fvg" / "ema_band"
    timeframe: str
    top: float
    bottom: float
    created_at: str = ""       # ISO timestamp of the origin bar
    tested: bool = False
    broken: bool = False
    strength: float = 0.0      # impulse strength that produced it

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2


class Level(BaseModel):
    """A horizontal support/resistance level built from clustered pivots."""
    price: float
    kind: str                  # "support" / "resistance"
    touches: int = 1
    timeframes: list[str] = []
    last_touch: str = ""


class TimeframeRead(BaseModel):
    """What one timeframe says about the chart."""
    timeframe: str
    label: str
    bars: int
    price: float
    trend: str                 # "uptrend" / "downtrend" / "range"
    trend_score: float         # -100..+100
    structure: str             # "HH/HL", "LH/LL", "mixed"
    last_event: str = ""       # "BOS up", "CHoCH down", ""
    ema_stack: str = ""        # "bullish" / "bearish" / "mixed"
    ema_20: float | None = None
    ema_50: float | None = None
    ema_200: float | None = None
    rsi_14: float | None = None
    macd_signal: str | None = None
    adx_14: float | None = None
    atr: float | None = None
    atr_pct: float | None = None
    rel_volume: float | None = None
    swing_high: float | None = None
    swing_low: float | None = None
    notes: list[str] = []


class EntrySetup(BaseModel):
    """A concrete, actionable entry plan."""
    direction: Direction = Direction.NONE
    setup_type: str = ""
    status: SetupStatus = SetupStatus.NO_SETUP
    entry_low: float | None = None
    entry_high: float | None = None
    entry_ref: float | None = None      # the price to work the order at
    stop: float | None = None
    target_1: float | None = None
    target_2: float | None = None
    target_3: float | None = None
    rr_target_1: float | None = None
    rr_target_2: float | None = None
    risk_per_share: float | None = None
    distance_to_entry_pct: float | None = None
    confidence: float = 0.0             # 0-100
    grade: str = "-"                    # A+ / A / B / C / D
    confluences: list[str] = []
    triggers: list[str] = []
    invalidation: str = ""
    timeframe_of_entry: str = ""
    first_obstacle: float | None = None   # nearest level between entry and target 1
    notes: list[str] = []


class ChartAnalysis(BaseModel):
    """Full multi-timeframe read for one ticker."""
    ticker: str
    as_of: str
    price: float
    bias: Direction = Direction.NONE
    bias_score: float = 0.0             # -100..+100, timeframe-weighted
    bias_label: str = ""
    alignment_pct: float = 0.0          # how much of the weight agrees with the bias
    timeframes: list[TimeframeRead] = []
    setup: EntrySetup = EntrySetup()
    zones: list[Zone] = []
    levels: list[Level] = []
    warnings: list[str] = []
