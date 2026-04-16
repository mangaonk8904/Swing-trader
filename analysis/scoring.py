from datetime import date
from models import (
    TechnicalSnapshot,
    FundamentalData,
    InstitutionalData,
    SeekingAlphaData,
    StockScore,
    Signal,
)
from analysis.fundamentals import score_fundamentals
from analysis.institutional import score_institutional
from analysis.sa_scoring import score_seeking_alpha
from config import settings


def score_technical(snap: TechnicalSnapshot) -> float:
    """Score technical indicators on a 0-100 scale.

    RSI zone:
      30-50 (oversold recovery) = 25 pts
      50-70 (momentum)          = 15 pts
      extremes (<30 or >70)     = 0 pts

    MACD bullish crossover      = 25 pts
    Price above SMA-50          = 15 pts
    Price above SMA-200         = 10 pts
    Volume > 1.5x avg           = 10 pts
    """
    score = 0.0

    # RSI (max 25)
    if snap.rsi_14 is not None:
        if 30 <= snap.rsi_14 <= 50:
            score += 25
        elif 50 < snap.rsi_14 <= 70:
            score += 15

    # MACD (max 25)
    if snap.macd_signal == "bullish":
        score += 25

    # SMA-50 (max 15)
    if snap.above_sma_50:
        score += 15

    # SMA-200 (max 10)
    if snap.above_sma_200:
        score += 10

    # Volume surge (max 10)
    if snap.volume_vs_avg is not None and snap.volume_vs_avg > 1.5:
        score += 10

    return min(score, 100.0)


def score_stock(
    tech: TechnicalSnapshot | None = None,
    fund: FundamentalData | None = None,
    inst: InstitutionalData | None = None,
    sa: SeekingAlphaData | None = None,
) -> StockScore:
    """Combine all available data into a weighted composite score.

    Weights re-normalize when a component is missing so the score
    still uses the full 0-100 range.
    """
    ticker = (tech and tech.ticker) or (fund and fund.ticker) or (inst and inst.ticker) or (sa and sa.ticker) or "UNKNOWN"
    today = (tech and tech.date) or date.today()

    # Compute sub-scores for available data
    components: list[tuple[float, float]] = []  # (score, weight)

    tech_score = 0.0
    fund_score = 0.0
    inst_score = 0.0
    sa_score = 0.0

    if tech is not None:
        tech_score = score_technical(tech)
        components.append((tech_score, settings.technical_weight))

    if fund is not None:
        fund_score = score_fundamentals(fund)
        components.append((fund_score, settings.fundamental_weight))

    if inst is not None:
        inst_score = score_institutional(inst)
        components.append((inst_score, settings.institutional_weight))

    if sa is not None:
        sa_score = score_seeking_alpha(sa)
        components.append((sa_score, settings.sa_weight))

    # Weighted composite with re-normalization
    if components:
        total_weight = sum(w for _, w in components)
        composite = sum(s * (w / total_weight) for s, w in components)
    else:
        composite = 0.0

    # Determine signal
    if composite >= settings.strong_buy_threshold:
        signal = Signal.STRONG_BUY
    elif composite >= settings.buy_threshold:
        signal = Signal.BUY
    elif composite >= settings.neutral_threshold:
        signal = Signal.NEUTRAL
    else:
        signal = Signal.PASS

    # Entry / Stop / Target using ATR
    entry = tech.price if tech else None
    stop_loss = None
    target = None
    if tech and tech.atr_14 and entry:
        stop_loss = round(entry - settings.stop_loss_atr_mult * tech.atr_14, 2)
        target = round(entry + settings.target_atr_mult * tech.atr_14, 2)

    # Build notes
    notes_parts = []
    if tech and tech.rsi_14 is not None:
        if tech.rsi_14 < 30:
            notes_parts.append("RSI oversold")
        elif tech.rsi_14 > 70:
            notes_parts.append("RSI overbought")
    if fund and fund.revenue_growth_pct is not None and fund.revenue_growth_pct > 20:
        notes_parts.append(f"Strong rev growth {fund.revenue_growth_pct:+.1f}%")
    if inst and inst.net_institutional is not None and inst.net_institutional > 50:
        notes_parts.append(f"Heavy inst buying (net {inst.net_institutional})")
    if sa and sa.mean_score >= 4.5:
        notes_parts.append(f"SA Strong Buy ({sa.rating})")
    elif sa and sa.momentum >= 10:
        notes_parts.append(f"SA high momentum ({sa.momentum}/12)")

    return StockScore(
        ticker=ticker,
        date=today,
        technical_score=round(tech_score, 1),
        fundamental_score=round(fund_score, 1),
        institutional_score=round(inst_score, 1),
        sa_score=round(sa_score, 1),
        composite_score=round(composite, 1),
        signal=signal,
        entry_price=entry,
        stop_loss=stop_loss,
        target_price=target,
        notes="; ".join(notes_parts),
    )
