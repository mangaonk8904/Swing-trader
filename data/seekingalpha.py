import httpx
import json
from pathlib import Path
from datetime import date
from config import settings

CACHE_DIR = Path(".cache/seekingalpha")
BASE_URL = "https://seeking-alpha.p.rapidapi.com"

# Seeking Alpha factor grade categories: 1 (D-) to 12 (A+)
GRADE_MAP = {
    12: "A+", 11: "A", 10: "A-",
    9: "B+", 8: "B", 7: "B-",
    6: "C+", 5: "C", 4: "C-",
    3: "D+", 2: "D", 1: "D-",
}


class SeekingAlphaClient:
    """Client for Seeking Alpha API via RapidAPI with file-based caching."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.seeking_alpha_rapidapi_key
        self.enabled = bool(self.api_key)
        if self.enabled:
            self.client = httpx.Client(
                headers={
                    "x-rapidapi-host": "seeking-alpha.p.rapidapi.com",
                    "x-rapidapi-key": self.api_key,
                },
                timeout=15.0,
            )

    def _cache_path(self, key: str, endpoint: str) -> Path:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        safe_endpoint = endpoint.replace("/", "_")
        return CACHE_DIR / f"{key}_{safe_endpoint}_{date.today()}.json"

    def _get_cached(self, key: str, endpoint: str) -> dict | None:
        path = self._cache_path(key, endpoint)
        if path.exists():
            return json.loads(path.read_text())
        return None

    def _set_cache(self, key: str, endpoint: str, data: dict):
        path = self._cache_path(key, endpoint)
        path.write_text(json.dumps(data))

    def _fetch(self, endpoint: str, params: dict, cache_key: str) -> dict | None:
        if not self.enabled:
            return None

        cached = self._get_cached(cache_key, endpoint)
        if cached is not None:
            return cached

        url = f"{BASE_URL}/{endpoint}"
        try:
            resp = self.client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            self._set_cache(cache_key, endpoint, data)
            return data
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                print(f"[SeekingAlpha] Access denied — check API key")
            elif e.response.status_code == 429:
                print(f"[SeekingAlpha] Rate limited — try again later")
            else:
                print(f"[SeekingAlpha] HTTP {e.response.status_code}")
            return None
        except httpx.RequestError as e:
            print(f"[SeekingAlpha] Request error: {e}")
            return None

    def get_grades(self, tickers: list[str]) -> dict[str, dict]:
        """Fetch factor grades for multiple tickers in one call.

        Returns dict mapping ticker -> {
            "value": int, "growth": int, "momentum": int,
            "profitability": int, "eps_revisions": int,
            "value_grade": str, "growth_grade": str, ...
        }
        """
        if not self.enabled or not tickers:
            return {}

        slugs = ",".join(t.lower() for t in tickers)
        cache_key = slugs.replace(",", "_")
        data = self._fetch("metrics-grades", {"slugs": slugs}, cache_key)
        if data is None:
            return {}

        results = {}
        grades_list = data.get("metrics_grades") or data.get("data") or []
        if isinstance(data, list):
            grades_list = data

        for entry in grades_list:
            ticker = (entry.get("slug") or "").upper()
            if not ticker:
                continue
            val = entry.get("value_category") or 0
            growth = entry.get("growth_category") or 0
            momentum = entry.get("momentum_category") or 0
            profit = entry.get("profitability_category") or 0
            eps_rev = entry.get("eps_revisions_category") or 0

            results[ticker] = {
                "value": val,
                "growth": growth,
                "momentum": momentum,
                "profitability": profit,
                "eps_revisions": eps_rev,
                "value_grade": GRADE_MAP.get(val, "N/A"),
                "growth_grade": GRADE_MAP.get(growth, "N/A"),
                "momentum_grade": GRADE_MAP.get(momentum, "N/A"),
                "profitability_grade": GRADE_MAP.get(profit, "N/A"),
                "eps_revisions_grade": GRADE_MAP.get(eps_rev, "N/A"),
            }

        return results

    def get_recommendations(self, tickers: list[str]) -> dict[str, dict]:
        """Fetch Wall Street analyst recommendations for multiple tickers.

        Returns dict mapping ticker -> {
            "analyst_count": int, "mean_score": float,
            "rating": str, "authors_count": int
        }
        """
        if not self.enabled or not tickers:
            return {}

        slugs = ",".join(t.lower() for t in tickers)
        cache_key = slugs.replace(",", "_")
        data = self._fetch("analyst-recommendation", {"slugs": slugs}, cache_key)
        if data is None:
            return {}

        results = {}
        recs_list = data.get("analyst_recommendation") or data.get("data") or []
        if isinstance(data, list):
            recs_list = data

        for entry in recs_list:
            ticker = (entry.get("slug") or "").upper()
            if not ticker:
                continue
            mean_score = entry.get("mean_score") or 0.0
            rating = _score_to_rating(mean_score)

            results[ticker] = {
                "analyst_count": entry.get("tot_analysts_recommendations") or 0,
                "mean_score": round(mean_score, 2),
                "rating": rating,
                "authors_count": entry.get("authors_count") or 0,
            }

        return results

    def get_ticker_data(self, tickers: list[str]) -> dict[str, dict]:
        """Fetch both grades and recommendations, merged per ticker."""
        grades = self.get_grades(tickers)
        recs = self.get_recommendations(tickers)

        merged = {}
        all_tickers = set(list(grades.keys()) + list(recs.keys()))
        for ticker in all_tickers:
            merged[ticker] = {
                **(grades.get(ticker) or {}),
                **(recs.get(ticker) or {}),
            }
        return merged


def _score_to_rating(score: float) -> str:
    """Convert 1-5 analyst mean score to a rating string."""
    if score >= 4.5:
        return "Strong Buy"
    elif score >= 3.5:
        return "Buy"
    elif score >= 2.5:
        return "Hold"
    elif score >= 1.5:
        return "Sell"
    elif score > 0:
        return "Strong Sell"
    return "N/A"
