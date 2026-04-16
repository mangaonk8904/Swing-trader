from models import SeekingAlphaData


def score_seeking_alpha(sa: SeekingAlphaData) -> float:
    """Score Seeking Alpha signals on a 0-100 scale.

    Factor grades (1-12 scale, each worth up to ~12 pts):
      Momentum grade     → max 15 pts  (most relevant for swing trading)
      EPS Revisions      → max 15 pts  (estimate momentum = price catalyst)
      Growth grade        → max 12 pts
      Profitability grade → max 10 pts
      Value grade         → max 8 pts   (less weight — swing ≠ value investing)

    Wall Street consensus (1-5 scale):
      Strong Buy (≥4.5)  → 40 pts
      Buy (≥3.5)         → 30 pts
      Hold (≥2.5)        → 15 pts
      Sell / Strong Sell  → 0 pts
    """
    score = 0.0

    # Momentum (max 15) — most important for swing trading
    score += min((sa.momentum / 12) * 15, 15.0)

    # EPS Revisions (max 15) — analyst estimate changes drive price
    score += min((sa.eps_revisions / 12) * 15, 15.0)

    # Growth (max 12)
    score += min((sa.growth / 12) * 12, 12.0)

    # Profitability (max 10)
    score += min((sa.profitability / 12) * 10, 10.0)

    # Value (max 8)
    score += min((sa.value / 12) * 8, 8.0)

    # Wall Street consensus (max 40)
    if sa.mean_score >= 4.5:
        score += 40
    elif sa.mean_score >= 3.5:
        score += 30
    elif sa.mean_score >= 2.5:
        score += 15
    # else 0

    return min(score, 100.0)
