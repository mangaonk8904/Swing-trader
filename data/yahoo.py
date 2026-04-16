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
