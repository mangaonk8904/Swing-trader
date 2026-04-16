from schemas import InstitutionalData


def score_institutional(data: InstitutionalData) -> float:
    """Score institutional/13F data on a 0-100 scale.

    Net institutional buyers:
      >50   = 40 pts
      10-50 = 25 pts
      0-10  = 10 pts
      net sellers = 0 pts

    Short interest:
      declining        = 20 pts (if short_interest_change < 0)
      <5%              = 20 pts
      5-15%            = 0 pts
      >15%             = -10 pts (penalty, floored at 0)
    """
    score = 0.0

    # Net institutional buyers component (max 40)
    if data.net_institutional is not None:
        if data.net_institutional > 50:
            score += 40
        elif data.net_institutional > 10:
            score += 25
        elif data.net_institutional > 0:
            score += 10

    # Short interest level (max 20)
    if data.short_interest_pct is not None:
        if data.short_interest_pct < 5:
            score += 20
        elif data.short_interest_pct > 15:
            score -= 10

    # Short interest trend (max 20)
    if data.short_interest_change is not None and data.short_interest_change < 0:
        score += 20

    # Bonus: very heavy institutional accumulation (max 20)
    if data.net_institutional is not None and data.net_institutional > 100:
        score += 20

    return max(min(score, 100.0), 0.0)
