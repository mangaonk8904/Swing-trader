import httpx
import json
from pathlib import Path
from datetime import date
from models import InstitutionalData
from config import settings

CACHE_DIR = Path(".cache/fintel")
BASE_URL = "https://api.fintel.io/web/v/0.0"


class FintelClient:
    """Client for Fintel.io REST API with file-based caching."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.fintel_api_key
        self.enabled = bool(self.api_key)
        if self.enabled:
            self.client = httpx.Client(
                headers={"X-API-KEY": self.api_key},
                timeout=15.0,
            )

    def _cache_path(self, ticker: str, endpoint: str) -> Path:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        safe_endpoint = endpoint.replace("/", "_")
        return CACHE_DIR / f"{ticker}_{safe_endpoint}_{date.today()}.json"

    def _get_cached(self, ticker: str, endpoint: str) -> dict | None:
        path = self._cache_path(ticker, endpoint)
        if path.exists():
            return json.loads(path.read_text())
        return None

    def _set_cache(self, ticker: str, endpoint: str, data: dict):
        path = self._cache_path(ticker, endpoint)
        path.write_text(json.dumps(data))

    def _fetch(self, endpoint: str, ticker: str) -> dict | None:
        if not self.enabled:
            return None

        # Check cache first
        cached = self._get_cached(ticker, endpoint)
        if cached is not None:
            return cached

        url = f"{BASE_URL}/{endpoint}/{ticker.lower()}"
        try:
            resp = self.client.get(url)
            resp.raise_for_status()
            data = resp.json()
            self._set_cache(ticker, endpoint, data)
            return data
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                print(f"[Fintel] Access denied for {ticker} — check API key/plan")
            elif e.response.status_code == 429:
                print(f"[Fintel] Rate limited — try again later")
            else:
                print(f"[Fintel] HTTP {e.response.status_code} for {ticker}")
            return None
        except httpx.RequestError as e:
            print(f"[Fintel] Request error for {ticker}: {e}")
            return None

    def _fetch_with_country(self, endpoint: str, ticker: str, country: str = "us") -> dict | None:
        """Fetch from endpoints that require a country segment (e.g. sf/us/aapl)."""
        if not self.enabled:
            return None

        cache_key = f"{endpoint}/{country}"
        cached = self._get_cached(ticker, cache_key)
        if cached is not None:
            return cached

        url = f"{BASE_URL}/{endpoint}/{country}/{ticker.lower()}"
        try:
            resp = self.client.get(url)
            resp.raise_for_status()
            data = resp.json()
            self._set_cache(ticker, cache_key, data)
            return data
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                print(f"[Fintel] Access denied for {endpoint}/{ticker} — check API key/plan")
            elif e.response.status_code == 429:
                print(f"[Fintel] Rate limited — try again later")
            else:
                print(f"[Fintel] HTTP {e.response.status_code} for {endpoint}/{ticker}")
            return None
        except httpx.RequestError as e:
            print(f"[Fintel] Request error for {endpoint}/{ticker}: {e}")
            return None

    def get_sec_filings(self, ticker: str) -> list[dict]:
        """Fetch recent SEC filings (10-K, 10-Q, 8-K, etc.)."""
        data = self._fetch_with_country("sf", ticker)
        if data is None:
            return []
        # Response may be a list or have a key containing the list
        if isinstance(data, list):
            return data
        # Try common wrapper keys
        for key in ("filings", "data", "results", "rows"):
            if key in data and isinstance(data[key], list):
                return data[key]
        return [data] if data else []

    def get_insider_trades(self, ticker: str) -> list[dict]:
        """Fetch recent insider trades (Form 3, 4, 5)."""
        data = self._fetch_with_country("n", ticker)
        if data is None:
            return []
        if isinstance(data, list):
            return data
        for key in ("trades", "data", "results", "rows"):
            if key in data and isinstance(data[key], list):
                return data[key]
        return [data] if data else []

    def get_institutional_ownership(self, ticker: str) -> list[dict]:
        """Fetch 13F institutional ownership filings (who owns what)."""
        data = self._fetch_with_country("so", ticker)
        if data is None:
            return []
        if isinstance(data, list):
            return data
        for key in ("owners", "holdings", "data", "results", "rows"):
            if key in data and isinstance(data[key], list):
                return data[key]
        return [data] if data else []

    def get_institutional_holdings(self, ticker: str) -> dict | None:
        """Fetch institutional ownership summary (13F data)."""
        return self._fetch("if", ticker)

    def get_short_interest(self, ticker: str) -> dict | None:
        """Fetch short interest data."""
        return self._fetch("ss", ticker)

    def get_institutional_data(self, ticker: str) -> InstitutionalData | None:
        """Fetch and combine institutional + short interest into InstitutionalData model."""
        if not self.enabled:
            return None

        inst_raw = self.get_institutional_holdings(ticker)
        short_raw = self.get_short_interest(ticker)

        if inst_raw is None and short_raw is None:
            return None

        buyers = None
        sellers = None
        net = None
        short_pct = None
        short_change = None

        # Parse institutional holdings
        if inst_raw:
            # Fintel returns ownership data — extract buyer/seller counts
            # The exact structure depends on the API tier
            buyers = _safe_int(inst_raw.get("newBuyers") or inst_raw.get("buyers"))
            sellers = _safe_int(inst_raw.get("newSellers") or inst_raw.get("sellers"))
            if buyers is not None and sellers is not None:
                net = buyers - sellers
            # Fallback: try net change fields
            if net is None:
                net = _safe_int(inst_raw.get("netInstitutional") or inst_raw.get("ownerCountChange"))

        # Parse short interest
        if short_raw:
            short_pct = _safe_float(short_raw.get("shortInterestPct") or short_raw.get("shortInterest"))
            short_change = _safe_float(short_raw.get("shortInterestChange") or short_raw.get("change"))

        return InstitutionalData(
            ticker=ticker.upper(),
            institutional_buyers=buyers,
            institutional_sellers=sellers,
            net_institutional=net,
            short_interest_pct=short_pct,
            short_interest_change=short_change,
        )


def _safe_float(val) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_int(val) -> int | None:
    if val is None:
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None
