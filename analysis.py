"""
analysis.py
Calculates technical indicators from a pandas DataFrame of OHLCV data.
Expected columns: open, high, low, close, volume  (all lowercase)
"""

import math
import pandas as pd
import numpy as np


def _clean(value):
    """Convert NaN / Inf / -Inf to None so the value is JSON-serialisable."""
    if value is None:
        return None
    try:
        if math.isnan(value) or math.isinf(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _clean_list(lst: list) -> list:
    """Apply _clean() to every element of a list."""
    return [_clean(v) for v in lst]


# ---------------------------------------------------------------------------
# Moving Averages
# ---------------------------------------------------------------------------

def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period, min_periods=1).mean().round(4)


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean().round(4)


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (0–100)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    return result.round(2)


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """
    Returns a dict of three Series:
        macd_line, signal_line, histogram
    """
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = (ema_fast - ema_slow).round(4)
    signal_line = ema(macd_line, signal)
    histogram = (macd_line - signal_line).round(4)
    return {"macd": macd_line, "signal": signal_line, "histogram": histogram}


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    """Returns upper, middle, and lower Bollinger Bands."""
    middle = sma(series, period)
    rolling_std = series.rolling(window=period, min_periods=1).std()
    upper = (middle + std_dev * rolling_std).round(4)
    lower = (middle - std_dev * rolling_std).round(4)
    return {"upper": upper, "middle": middle, "lower": lower}


# ---------------------------------------------------------------------------
# Main builder – returns a ready-to-use dict for the API
# ---------------------------------------------------------------------------

def build_indicators(df: pd.DataFrame) -> dict:
    """
    Given a DataFrame with columns [date, open, high, low, close, volume],
    returns a dict containing all indicator series as lists (JSON-serialisable).
    """
    close = df["close"]
    dates = df["date"].astype(str).tolist()

    sma_20 = sma(close, 20)
    sma_50 = sma(close, 50)
    sma_200 = sma(close, 200)
    rsi_14 = rsi(close, 14)
    macd_data = macd(close)
    bb = bollinger_bands(close, 20)

    # Determine RSI signal (guard against NaN on short series)
    _raw_rsi = rsi_14.iloc[-1] if not rsi_14.empty else 50
    latest_rsi = _clean(_raw_rsi) or 50.0
    if latest_rsi < 30:
        rsi_signal = "Oversold"
    elif latest_rsi > 70:
        rsi_signal = "Overbought"
    else:
        rsi_signal = "Neutral"

    latest_rsi_clean = _clean(float(latest_rsi)) or 50.0

    return {
        "dates": dates,
        "close":  _clean_list(close.round(2).tolist()),
        "open":   _clean_list(df["open"].round(2).tolist()),
        "high":   _clean_list(df["high"].round(2).tolist()),
        "low":    _clean_list(df["low"].round(2).tolist()),
        "volume": _clean_list(df["volume"].tolist()),
        "sma_20":  _clean_list(sma_20.tolist()),
        "sma_50":  _clean_list(sma_50.tolist()),
        "sma_200": _clean_list(sma_200.tolist()),
        "rsi":     _clean_list(rsi_14.tolist()),
        "rsi_signal": rsi_signal,
        "latest_rsi": round(latest_rsi_clean, 2),
        "macd":           _clean_list(macd_data["macd"].tolist()),
        "macd_signal":    _clean_list(macd_data["signal"].tolist()),
        "macd_histogram": _clean_list(macd_data["histogram"].tolist()),
        "bb_upper":  _clean_list(bb["upper"].tolist()),
        "bb_middle": _clean_list(bb["middle"].tolist()),
        "bb_lower":  _clean_list(bb["lower"].tolist()),
    }
