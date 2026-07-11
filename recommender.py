"""
recommender.py
Scores a list of tickers and returns the top recommended stocks.

Scoring factors (each 0-100, weighted):
  - Predicted 30-day % gain      (weight 40%)
  - RSI health (40-60 is ideal)  (weight 25%)
  - SMA trend (price > SMA50)    (weight 20%)
  - MACD signal                  (weight 15%)
"""

import yfinance as yf
import pandas as pd
import numpy as np

from analysis import build_indicators
from predictor import predict

# Default pool of popular stocks to evaluate
DEFAULT_POOL = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "TSLA", "META", "JPM", "V", "UNH",
    "JNJ", "WMT", "PG", "MA", "HD",
    "INFY", "TCS.NS", "RELIANCE.NS", "HDFCBANK.NS", "WIPRO.NS",
]


def _fetch_df(ticker: str) -> pd.DataFrame | None:
    """Download ~2 years of daily data for a ticker. Returns None on failure."""
    try:
        raw = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
        if raw.empty or len(raw) < 60:
            return None
        raw = raw.reset_index()
        raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in raw.columns]
        raw = raw.rename(columns={"price": "close"}) if "price" in raw.columns else raw
        # yfinance sometimes returns multi-level columns — flatten
        if hasattr(raw.columns, "levels"):
            raw.columns = ["_".join(filter(None, c)).lower() for c in raw.columns]
        raw = raw[["date", "open", "high", "low", "close", "volume"]].dropna()
        raw["date"] = pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d")
        return raw
    except Exception:
        return None


def _score_stock(ticker: str) -> dict | None:
    """Return a score dict for a single ticker or None if data unavailable."""
    df = _fetch_df(ticker)
    if df is None:
        return None

    # --- Indicators ---
    try:
        indicators = build_indicators(df)
    except Exception:
        return None

    # --- Prediction ---
    try:
        pred_result = predict(df)
        if pred_result.get("error") or not pred_result.get("predictions"):
            return None
    except Exception:
        return None

    # Extract 30-day prediction
    pred_30 = next((p for p in pred_result["predictions"] if p["horizon_days"] == 30), None)
    if pred_30 is None:
        return None

    pct_change_30 = pred_30["pct_change"]
    trend = pred_30["trend"]

    # --- Sub-scores ---
    # 1. Predicted gain score (capped at ±20%)
    gain_score = min(max((pct_change_30 + 20) / 40 * 100, 0), 100)

    # 2. RSI health score (ideal zone 40-60)
    latest_rsi = indicators["latest_rsi"]
    if 40 <= latest_rsi <= 60:
        rsi_score = 100
    elif latest_rsi < 30 or latest_rsi > 70:
        rsi_score = 20
    else:
        distance = min(abs(latest_rsi - 50), 20)
        rsi_score = 100 - (distance / 20) * 80

    # 3. SMA trend score (price vs SMA50)
    current_price = indicators["close"][-1]
    sma50_vals = [v for v in indicators["sma_50"] if v is not None and v > 0]
    sma50 = sma50_vals[-1] if sma50_vals else current_price
    sma_score = 100 if current_price > sma50 else 30

    # 4. MACD signal score
    macd_hist = indicators["macd_histogram"]
    recent_hist = [v for v in macd_hist[-5:] if v is not None]
    if recent_hist and recent_hist[-1] > 0:
        macd_score = 80 if recent_hist[-1] > (recent_hist[0] if recent_hist[0] else 0) else 60
    else:
        macd_score = 20

    # Weighted total
    total_score = (
        gain_score * 0.40
        + rsi_score * 0.25
        + sma_score * 0.20
        + macd_score * 0.15
    )

    # Risk level
    if total_score >= 70:
        risk = "Low"
    elif total_score >= 45:
        risk = "Medium"
    else:
        risk = "High"

    # Reason string
    reasons = []
    if pct_change_30 > 5:
        reasons.append(f"Strong {pct_change_30:+.1f}% predicted 30-day gain")
    elif pct_change_30 < -5:
        reasons.append(f"Predicted decline of {pct_change_30:.1f}%")
    if 40 <= latest_rsi <= 60:
        reasons.append("RSI in healthy zone")
    elif latest_rsi < 30:
        reasons.append("RSI oversold – potential bounce")
    if current_price > sma50:
        reasons.append("Price above 50-day SMA (uptrend)")
    if not reasons:
        reasons.append("Stable trend with moderate outlook")

    return {
        "ticker": ticker,
        "score": round(total_score, 1),
        "trend": trend,
        "predicted_price_30d": pred_30["predicted_price"],
        "pct_change_30d": pct_change_30,
        "rsi": latest_rsi,
        "risk": risk,
        "reason": "; ".join(reasons),
        "current_price": round(current_price, 2),
    }


def get_recommendations(pool: list[str] = None, top_n: int = 5) -> list[dict]:
    """
    Score each ticker in the pool and return the top_n results sorted by score.
    Skips any ticker that fails to load or predict.
    """
    if pool is None:
        pool = DEFAULT_POOL

    results = []
    for ticker in pool:
        scored = _score_stock(ticker)
        if scored:
            results.append(scored)

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]
