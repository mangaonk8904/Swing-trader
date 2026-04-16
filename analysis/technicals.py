import pandas as pd
import pandas_ta as ta
from datetime import date
from schemas import TechnicalSnapshot
from config import settings


def compute_technicals(ticker: str, df: pd.DataFrame) -> TechnicalSnapshot:
    """Compute swing trading technical indicators from OHLCV data.

    Returns a TechnicalSnapshot with RSI, MACD, SMA positions, volume ratio, and ATR.
    """
    if len(df) < settings.sma_long:
        # Not enough data for SMA-200, compute what we can
        pass

    close = df["Close"]
    price = float(close.iloc[-1])
    today = df.index[-1]
    if hasattr(today, "date"):
        today = today.date()

    # RSI
    rsi_series = ta.rsi(close, length=settings.rsi_period)
    rsi_val = float(rsi_series.iloc[-1]) if rsi_series is not None and not rsi_series.empty else None

    # MACD
    macd_df = ta.macd(close, fast=12, slow=26, signal=9)
    macd_signal_str = None
    if macd_df is not None and not macd_df.empty:
        macd_line = macd_df.iloc[-1, 0]  # MACD line
        signal_line = macd_df.iloc[-1, 2]  # Signal line
        if macd_line > signal_line:
            macd_signal_str = "bullish"
        elif macd_line < signal_line:
            macd_signal_str = "bearish"
        else:
            macd_signal_str = "neutral"

    # SMAs
    sma_50 = ta.sma(close, length=settings.sma_short)
    above_sma_50 = None
    if sma_50 is not None and not sma_50.empty:
        above_sma_50 = price > float(sma_50.iloc[-1])

    sma_200 = ta.sma(close, length=settings.sma_long)
    above_sma_200 = None
    if sma_200 is not None and not sma_200.empty and not pd.isna(sma_200.iloc[-1]):
        above_sma_200 = price > float(sma_200.iloc[-1])

    # Volume vs 20-day average
    vol = df["Volume"]
    vol_avg = vol.rolling(window=settings.volume_avg_period).mean()
    volume_ratio = None
    if not vol_avg.empty and not pd.isna(vol_avg.iloc[-1]) and vol_avg.iloc[-1] > 0:
        volume_ratio = round(float(vol.iloc[-1]) / float(vol_avg.iloc[-1]), 2)

    # ATR
    atr_series = ta.atr(df["High"], df["Low"], close, length=settings.atr_period)
    atr_val = None
    if atr_series is not None and not atr_series.empty and not pd.isna(atr_series.iloc[-1]):
        atr_val = round(float(atr_series.iloc[-1]), 2)

    return TechnicalSnapshot(
        ticker=ticker,
        date=today,
        price=round(price, 2),
        rsi_14=round(rsi_val, 2) if rsi_val is not None else None,
        macd_signal=macd_signal_str,
        above_sma_50=above_sma_50,
        above_sma_200=above_sma_200,
        volume_vs_avg=volume_ratio,
        atr_14=atr_val,
    )
