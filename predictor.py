"""
predictor.py
Predicts stock prices at 7, 30, and 90-day horizons.

Strategy: Train one Ridge Regression model per horizon.
Each model predicts close[t + horizon] directly from features at time t.
This avoids error accumulation from iterative single-step forecasting and
produces meaningfully different prices for each horizon.
"""

import math
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_percentage_error


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a rich feature matrix from OHLCV data.
    Returns a new DataFrame; the caller drops NaNs after calling this.
    """
    df = df.copy()
    c = df["close"].astype(float)
    v = df["volume"].astype(float)

    # ── Lag features (recent prices) ────────────────────────────────────────
    for lag in [1, 2, 3, 5, 7, 10, 14, 20, 30]:
        df[f"lag_{lag}"] = c.shift(lag)

    # ── Returns (% change) ──────────────────────────────────────────────────
    for lag in [1, 3, 5, 10, 20]:
        df[f"ret_{lag}"] = c.pct_change(lag)

    # ── Rolling statistics ───────────────────────────────────────────────────
    for w in [5, 10, 20, 50, 100]:
        df[f"roll_mean_{w}"]  = c.rolling(w).mean()
        df[f"roll_std_{w}"]   = c.rolling(w).std()
        df[f"roll_min_{w}"]   = c.rolling(w).min()
        df[f"roll_max_{w}"]   = c.rolling(w).max()

    # ── Momentum & trend ────────────────────────────────────────────────────
    for p in [5, 10, 20, 30]:
        df[f"mom_{p}"]   = c - c.shift(p)
        df[f"mom_r_{p}"] = c / c.shift(p).replace(0, np.nan)

    # ── Exponential moving averages ──────────────────────────────────────────
    for span in [12, 26, 50]:
        df[f"ema_{span}"] = c.ewm(span=span, adjust=False).mean()

    # ── EMA spread (trend direction signal) ─────────────────────────────────
    df["ema_diff_12_26"] = df["ema_12"] - df["ema_26"]
    df["ema_diff_26_50"] = df["ema_26"] - df["ema_50"]
    df["price_vs_ema50"] = c - df["ema_50"]

    # ── Volatility ───────────────────────────────────────────────────────────
    df["atr_14"] = (
        df["high"].astype(float) - df["low"].astype(float)
    ).rolling(14).mean()

    # ── Volume features ──────────────────────────────────────────────────────
    df["vol_change"]  = v.pct_change(1)
    df["vol_ma_10"]   = v.rolling(10).mean()
    df["vol_ratio"]   = v / df["vol_ma_10"].replace(0, np.nan)

    # ── RSI (14) ─────────────────────────────────────────────────────────────
    delta   = c.diff()
    gain    = delta.clip(lower=0).ewm(com=13, min_periods=14).mean()
    loss    = (-delta.clip(upper=0)).ewm(com=13, min_periods=14).mean()
    rs      = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # ── Calendar features ────────────────────────────────────────────────────
    dates = pd.to_datetime(df["date"])
    df["day_of_week"] = dates.dt.dayofweek
    df["month"]       = dates.dt.month
    df["day_sin"]     = np.sin(2 * np.pi * dates.dt.dayofyear / 365)
    df["day_cos"]     = np.cos(2 * np.pi * dates.dt.dayofyear / 365)

    return df


FEATURE_COLS = None   # resolved lazily on first call


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {"date", "open", "high", "low", "close", "volume"}
    return [c for c in df.columns if c not in exclude]


# ---------------------------------------------------------------------------
# Per-horizon model training
# ---------------------------------------------------------------------------

def _train_horizon_model(df: pd.DataFrame, horizon: int):
    """
    Train a Ridge model that predicts close[t + horizon] from features at t.

    Returns (model, scaler, feature_cols, mape) or raises on failure.
    """
    feat_df = _engineer_features(df)
    feat_df = feat_df.dropna()

    feature_cols = _get_feature_cols(feat_df)
    X = feat_df[feature_cols].values.astype(float)
    y = feat_df["close"].values.astype(float)

    # Target: price `horizon` steps ahead
    # Align: X[i] → y[i + horizon]
    if len(X) < horizon + 60:
        raise ValueError(f"Not enough data to train {horizon}-day model")

    X_aligned = X[:len(X) - horizon]
    y_aligned  = y[horizon:]

    # Train/test split (no shuffle — time series)
    split = int(len(X_aligned) * 0.8)
    X_train, X_test = X_aligned[:split], X_aligned[split:]
    y_train, y_test = y_aligned[:split],  y_aligned[split:]

    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = Ridge(alpha=1.0)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    # Guard against zero targets causing inf MAPE
    mask = y_test != 0
    mape = mean_absolute_percentage_error(y_test[mask], y_pred[mask]) if mask.sum() > 0 else 0.1

    return model, scaler, feature_cols, mape


# ---------------------------------------------------------------------------
# Forecast helpers
# ---------------------------------------------------------------------------

def _safe(v):
    if v is None:
        return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _trend_label(current: float, predicted: float) -> str:
    if current == 0:
        return "Neutral"
    pct = (predicted - current) / current * 100
    if pct > 3:
        return "Bullish"
    elif pct < -3:
        return "Bearish"
    return "Neutral"


def _confidence_from_mape(mape: float) -> float:
    """Higher MAPE → lower confidence. Cap at 95%."""
    conf = max(0.05, 1.0 - mape * 4)
    return round(min(conf, 0.95), 3)


# ---------------------------------------------------------------------------
# Main predict function
# ---------------------------------------------------------------------------

def predict(df: pd.DataFrame) -> dict:
    """
    Predict future stock prices at 7, 30, and 90-day horizons.

    Parameters
    ----------
    df : DataFrame with columns [date, open, high, low, close, volume]

    Returns
    -------
    dict containing predictions list, forecast_dates, forecast_prices, etc.
    """
    if len(df) < 120:
        return {
            "error": "Not enough historical data (need at least 120 days).",
            "predictions": [],
        }

    df = df.sort_values("date").reset_index(drop=True)
    df["close"]  = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    current_price = float(df["close"].iloc[-1])

    horizons = [7, 30, 90]
    predictions = []
    best_mape = None

    for horizon in horizons:
        try:
            model, scaler, feature_cols, mape = _train_horizon_model(df, horizon)
        except Exception as e:
            predictions.append({
                "horizon_days":    horizon,
                "predicted_price": round(current_price, 2),
                "pct_change":      0.0,
                "trend":           "Neutral",
                "confidence":      0.0,
            })
            continue

        # Predict using the LAST available row of features
        feat_df = _engineer_features(df).dropna()
        X_last  = feat_df[feature_cols].iloc[-1:].values.astype(float)

        # Replace any NaN/Inf in X_last with column medians from training
        col_medians = np.nanmedian(feat_df[feature_cols].values.astype(float), axis=0)
        for col_idx in range(X_last.shape[1]):
            if not np.isfinite(X_last[0, col_idx]):
                X_last[0, col_idx] = col_medians[col_idx]

        X_last_s = scaler.transform(X_last)
        raw_pred = float(model.predict(X_last_s)[0])

        # Clamp prediction to ±60% of current price (sanity bound)
        lower = current_price * 0.40
        upper = current_price * 1.60
        predicted_price = max(lower, min(upper, raw_pred))

        pct_change = round((predicted_price - current_price) / current_price * 100, 2)
        trend      = _trend_label(current_price, predicted_price)
        confidence = _confidence_from_mape(mape)

        if best_mape is None or mape < best_mape:
            best_mape = mape

        predictions.append({
            "horizon_days":    horizon,
            "predicted_price": round(predicted_price, 2),
            "pct_change":      pct_change,
            "trend":           trend,
            "confidence":      round(confidence * 100, 1),
        })

    # ── Build 30-day daily forecast series for the chart ─────────────────────
    # Use the 30-day model to estimate each future day iteratively,
    # but scale by the fraction of the horizon to give a smooth curve.
    forecast_dates  = []
    forecast_prices = []

    pred_30_price = next(
        (p["predicted_price"] for p in predictions if p["horizon_days"] == 30),
        current_price
    )
    pred_7_price = next(
        (p["predicted_price"] for p in predictions if p["horizon_days"] == 7),
        current_price
    )

    last_date = pd.to_datetime(df["date"].iloc[-1])

    for day in range(1, 31):
        future_date = last_date + pd.Timedelta(days=day)
        # Interpolate smoothly from current → 7-day → 30-day target
        if day <= 7:
            frac = day / 7
            price = current_price + frac * (pred_7_price - current_price)
        else:
            frac = (day - 7) / 23
            price = pred_7_price + frac * (pred_30_price - pred_7_price)

        # Add small noise proportional to historical volatility for realism
        hist_std = float(df["close"].pct_change().std()) * current_price
        np.random.seed(day)   # deterministic so chart doesn't jump on reload
        noise = np.random.normal(0, hist_std * 0.3)
        price = round(max(current_price * 0.5, price + noise), 2)

        forecast_dates.append(future_date.strftime("%Y-%m-%d"))
        forecast_prices.append(price)

    overall_trend = _trend_label(current_price, pred_30_price)
    avg_confidence = round(
        sum(p["confidence"] for p in predictions) / len(predictions), 1
    ) if predictions else 0.0

    return {
        "current_price":  round(current_price, 2),
        "predictions":    predictions,
        "confidence":     avg_confidence,
        "trend":          overall_trend,
        "mape":           round(best_mape * 100, 2) if best_mape is not None else None,
        "forecast_dates": forecast_dates,
        "forecast_prices": forecast_prices,
    }
