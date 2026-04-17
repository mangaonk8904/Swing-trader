import yfinance as yf
import pandas as pd


def get_price_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Fetch OHLCV price history from Yahoo Finance.

    Returns a DataFrame with columns: Open, High, Low, Close, Volume
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    if df.empty:
        raise ValueError(f"No price data found for {ticker}")
    return df


def get_options_expirations(ticker: str) -> list[str]:
    """Return available options expiration dates for a ticker."""
    stock = yf.Ticker(ticker)
    expirations = stock.options
    if not expirations:
        raise ValueError(f"No options data available for {ticker}")
    return list(expirations)


def get_options_chain(ticker: str, expiry: str | None = None) -> dict:
    """Fetch options chain for a specific expiration.

    Returns {"calls": DataFrame, "puts": DataFrame, "expiry": str}
    """
    stock = yf.Ticker(ticker)
    if expiry is None:
        expirations = stock.options
        if not expirations:
            raise ValueError(f"No options data available for {ticker}")
        expiry = expirations[0]

    chain = stock.option_chain(expiry)
    return {"calls": chain.calls, "puts": chain.puts, "expiry": expiry}


def get_all_options_summary(ticker: str, max_expiries: int = 6) -> dict:
    """Aggregate options metrics across nearest expirations.

    Returns summary with put/call ratios, volume, OI, and per-expiry breakdown.
    """
    stock = yf.Ticker(ticker)
    expirations = stock.options
    if not expirations:
        raise ValueError(f"No options data available for {ticker}")

    # Get current price
    info = stock.info
    current_price = info.get("currentPrice") or info.get("regularMarketPrice") or 0.0

    expirations = list(expirations[:max_expiries])

    total_call_vol = 0
    total_put_vol = 0
    total_call_oi = 0
    total_put_oi = 0
    by_expiry = []

    for exp in expirations:
        try:
            chain = stock.option_chain(exp)
            cv = int(chain.calls["volume"].fillna(0).sum())
            pv = int(chain.puts["volume"].fillna(0).sum())
            coi = int(chain.calls["openInterest"].fillna(0).sum())
            poi = int(chain.puts["openInterest"].fillna(0).sum())

            total_call_vol += cv
            total_put_vol += pv
            total_call_oi += coi
            total_put_oi += poi

            pc_ratio = round(pv / cv, 2) if cv > 0 else 0.0
            by_expiry.append({
                "expiry": exp,
                "call_volume": cv,
                "put_volume": pv,
                "call_oi": coi,
                "put_oi": poi,
                "pc_ratio": pc_ratio,
            })
        except Exception:
            continue

    pc_vol_ratio = round(total_put_vol / total_call_vol, 2) if total_call_vol > 0 else 0.0
    pc_oi_ratio = round(total_put_oi / total_call_oi, 2) if total_call_oi > 0 else 0.0

    return {
        "current_price": current_price,
        "expirations": expirations,
        "total_call_volume": total_call_vol,
        "total_put_volume": total_put_vol,
        "total_call_oi": total_call_oi,
        "total_put_oi": total_put_oi,
        "pc_volume_ratio": pc_vol_ratio,
        "pc_oi_ratio": pc_oi_ratio,
        "by_expiry": by_expiry,
    }


def get_basic_fundamentals(ticker: str) -> dict:
    """Fetch basic fundamental info from Yahoo Finance."""
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "name": info.get("shortName", ticker),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "market_cap": info.get("marketCap"),
        "pe_ratio": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "dividend_yield": info.get("dividendYield"),
        "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
        "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
    }
