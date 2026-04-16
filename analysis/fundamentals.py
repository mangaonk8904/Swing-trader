from models import FundamentalData


def score_fundamentals(data: FundamentalData) -> float:
    """Score fundamental data on a 0-100 scale.

    Revenue growth tiers:
      >20% YoY  = 40 pts
      10-20%    = 25 pts
      0-10%     = 10 pts
      declining = 0 pts

    Seeking Alpha rating:
      strong_buy = 35 pts
      buy        = 25 pts
      hold/worse = 0 pts
    """
    score = 0.0

    # Revenue growth component (max 40)
    if data.revenue_growth_pct is not None:
        if data.revenue_growth_pct > 20:
            score += 40
        elif data.revenue_growth_pct > 10:
            score += 25
        elif data.revenue_growth_pct > 0:
            score += 10

    # Seeking Alpha rating component (max 35)
    if data.seeking_alpha_rating:
        rating = data.seeking_alpha_rating.lower().strip()
        if rating in ("strong buy", "strong_buy"):
            score += 35
        elif rating == "buy":
            score += 25

    # Bonus: strong revenue acceleration (max 25)
    # If growth > 50%, extra points for momentum
    if data.revenue_growth_pct is not None and data.revenue_growth_pct > 50:
        score += 15
    elif data.revenue_growth_pct is not None and data.revenue_growth_pct > 30:
        score += 10

    return min(score, 100.0)
