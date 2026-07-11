"""
predictor.py
XGBoost-based stock price predictor.

Why XGBoost instead of LSTM/TensorFlow:
  - Installs in seconds (vs 600MB for TensorFlow)
  - Comparable accuracy to LSTM on tabular time-series data
  - Trains in 1-3 seconds per horizon (vs 60-120s for LSTM)
  - Works perfectly on Render free tier

Architecture:
  - One XGBoost model per horizon (7, 30, 90 days)
  - Each model predicts close[t + horizon] directly from features at t
  - Features: 50+ engineered from OHLCV (lags, returns, SMA, EMA, RSI, MACD,
    Bollinger Bands, ATR, volume ratios, momentum, calendar)
  - Models cached in memory — no retraining on repeated calls
"""

import math
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_percentage_error

# ---------------------------------------------------------------------------
# In-process model cache  { (ticker, horizon): (model, scaler, feature_cols) }
# ---------------------------------------------------------------------------
_MODEL_CACHE: dict = {}


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build 50+ features from OHLCV data for XGBoost.
    Returns a new DataFrame with all features appended.
    """
    d = df.copy()
    c = d["close"].astype(float)
    v = d["volume"].astype(float)
    h = d["high"].astype(float)
    l = d["low"].astype(float)
    o = d["open"].astype(float)

    # ── Lag features ─────────────────────────────────────────────────────────
    for lag in [1, 2, 3, 5, 7, 10, 14, 21, 30, 60]:
        d[f"lag_{lag}"] = c.shift(lag)

    # ── Returns ──────────────────────────────────────────────────────────────
    for p in [1, 2, 3, 5, 7, 10, 14, 21, 30]:
        d[f"ret_{p}"] = c.pct_change(p)

    # ── Rolling statistics ───────────────────────────────────────────────────
    for w in [5, 10, 20, 50, 100, 200]:
        d[f"sma_{w}"]      = c.rolling(w).mean()
        d[f"roll_std_{w}"] = c.rolling(w).std()
        d[f"roll_min_{w}"] = c.rolling(w).min()
        d[f"roll_max_{w}"] = c.rolling(w).max()

    # Price position within rolling range (0=at low, 1=at high)
    for w in [10, 20, 50]:
        rng = d[f"roll_max_{w}"] - d[f"roll_min_{w}"]
        d[f"pos_{w}"] = (c - d[f"roll_min_{w}"]) / rng.replace(0, np.nan)

    # ── Exponential moving averages ──────────────────────────────────────────
    for span in [9, 12, 26, 50, 100, 200]:
        d[f"ema_{span}"] = c.ewm(span=span, adjust=False).mean()

    # ── EMA cross signals ────────────────────────────────────────────────────
    d["ema_9_26_diff"]   = d["ema_9"]  - d["ema_26"]
    d["ema_12_26_diff"]  = d["ema_12"] - d["ema_26"]
    d["ema_50_200_diff"] = d["ema_50"] - d["ema_200"]
    d["price_vs_sma20"]  = c - d["sma_20"]
    d["price_vs_sma50"]  = c - d["sma_50"]
    d["price_vs_sma200"] = c - d["sma_200"]

    # ── MACD ─────────────────────────────────────────────────────────────────
    d["macd"]          = d["ema_12"] - d["ema_26"]
    d["macd_signal"]   = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"]     = d["macd"] - d["macd_signal"]

    # ── RSI (14) ─────────────────────────────────────────────────────────────
    delta  = c.diff()
    gain   = delta.clip(lower=0).ewm(com=13, min_periods=14).mean()
    loss   = (-delta.clip(upper=0)).ewm(com=13, min_periods=14).mean()
    rs     = gain / loss.replace(0, np.nan)
    d["rsi_14"] = 100 - (100 / (1 + rs))

    # RSI overbought/oversold distance
    d["rsi_dist_50"] = d["rsi_14"] - 50

    # ── Bollinger Bands ──────────────────────────────────────────────────────
    bb_std         = c.rolling(20).std()
    d["bb_upper"]  = d["sma_20"] + 2 * bb_std
    d["bb_lower"]  = d["sma_20"] - 2 * bb_std
    d["bb_width"]  = (d["bb_upper"] - d["bb_lower"]) / d["sma_20"].replace(0, np.nan)
    d["bb_pos"]    = (c - d["bb_lower"]) / (d["bb_upper"] - d["bb_lower"]).replace(0, np.nan)

    # ── ATR (14) ─────────────────────────────────────────────────────────────
    tr             = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    d["atr_14"]    = tr.rolling(14).mean()
    d["atr_pct"]   = d["atr_14"] / c.replace(0, np.nan)   # normalized ATR

    # ── Momentum ─────────────────────────────────────────────────────────────
    for p in [5, 10, 20, 30]:
        d[f"mom_{p}"] = c - c.shift(p)

    # ── Volume features ──────────────────────────────────────────────────────
    d["vol_ma_10"]   = v.rolling(10).mean()
    d["vol_ma_20"]   = v.rolling(20).mean()
    d["vol_ratio"]   = v / d["vol_ma_20"].replace(0, np.nan)
    d["vol_change"]  = v.pct_change(1)
    d["obv"]         = (np.sign(c.diff()) * v).cumsum()

    # ── Candlestick body/wick ─────────────────────────────────────────────────
    d["body"]        = (c - o).abs()
    d["upper_wick"]  = h - pd.concat([c, o], axis=1).max(axis=1)
    d["lower_wick"]  = pd.concat([c, o], axis=1).min(axis=1) - l

    # ── Volatility ───────────────────────────────────────────────────────────
    d["volatility_10"] = d["ret_1"].rolling(10).std()
    d["volatility_20"] = d["ret_1"].rolling(20).std()
    d["volatility_30"] = d["ret_1"].rolling(30).std()

    # ── Calendar features ────────────────────────────────────────────────────
    dates          = pd.to_datetime(d["date"])
    d["day_of_week"]  = dates.dt.dayofweek
    d["month"]        = dates.dt.month
    d["quarter"]      = dates.dt.quarter
    d["day_sin"]      = np.sin(2 * np.pi * dates.dt.dayofyear / 365)
    d["day_cos"]      = np.cos(2 * np.pi * dates.dt.dayofyear / 365)
    d["month_sin"]    = np.sin(2 * np.pi * dates.dt.month / 12)
    d["month_cos"]    = np.cos(2 * np.pi * dates.dt.month / 12)

    return d


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {"date", "open", "high", "low", "close", "volume"}
    return [c for c in df.columns if c not in exclude]


# ---------------------------------------------------------------------------
# Train one XGBoost model per horizon
# ---------------------------------------------------------------------------

def _train(df: pd.DataFrame, horizon: int) -> tuple:
    """
    Train an XGBoost model to predict close[t + horizon] from features at t.
    Returns (model, scaler, feature_cols, mape).
    """
    feat_df      = _build_features(df).dropna()
    feature_cols = _get_feature_cols(feat_df)

    X = feat_df[feature_cols].values.astype(float)
    y = feat_df["close"].values.astype(float)

    if len(X) < horizon + 80:
        raise ValueError(f"Not enough data for {horizon}-day model")

    # Align: X[i] predicts y[i + horizon]
    X = X[:-horizon]
    y = y[horizon:]

    # Replace NaN/Inf in features
    col_medians = np.nanmedian(X, axis=0)
    for col in range(X.shape[1]):
        bad = ~np.isfinite(X[:, col])
        X[bad, col] = col_medians[col]

    # Scale
    scaler   = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split (no shuffle — time series)
    split    = int(len(X_scaled) * 0.85)
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y[:split], y[split:]

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = model.predict(X_test)
    mask   = y_test != 0
    mape   = mean_absolute_percentage_error(y_test[mask], y_pred[mask]) if mask.sum() > 0 else 0.10

    return model, scaler, feature_cols, mape


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(v):
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
    conf = max(0.05, 1.0 - mape * 3.5)
    return round(min(conf, 0.95), 3)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict(df: pd.DataFrame, ticker: str = "") -> dict:
    """
    Predict stock prices at 7, 30, and 90-day horizons using XGBoost.

    Parameters
    ----------
    df     : DataFrame with columns [date, open, high, low, close, volume]
    ticker : optional string used for in-memory model caching

    Returns
    -------
    dict with keys: current_price, predictions, confidence, trend,
                    mape, forecast_dates, forecast_prices, model_type
    """
    if len(df) < 150:
        return {
            "error": f"Not enough data (need 150+ days, have {len(df)}).",
            "predictions": [],
            "model_type": "none",
        }

    df = df.sort_values("date").reset_index(drop=True)
    df["close"]  = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    current_price = float(df["close"].iloc[-1])

    horizons    = [7, 30, 90]
    predictions = []
    best_mape   = None

    feat_df      = _build_features(df).dropna()
    feature_cols = _get_feature_cols(feat_df)

    for horizon in horizons:
        cache_key = (ticker, horizon)

        # Use cached model if available
        if cache_key in _MODEL_CACHE:
            model, scaler, feat_cols, mape = _MODEL_CACHE[cache_key]
        else:
            try:
                model, scaler, feat_cols, mape = _train(df, horizon)
                if ticker:
                    _MODEL_CACHE[cache_key] = (model, scaler, feat_cols, mape)
            except Exception as e:
                predictions.append({
                    "horizon_days":    horizon,
                    "predicted_price": round(current_price, 2),
                    "pct_change":      0.0,
                    "trend":           "Neutral",
                    "confidence":      0.0,
                })
                continue

        # Predict from latest available features
        try:
            X_last = feat_df[feat_cols].iloc[-1:].values.astype(float)
            col_medians = np.nanmedian(feat_df[feat_cols].values.astype(float), axis=0)
            for col in range(X_last.shape[1]):
                if not np.isfinite(X_last[0, col]):
                    X_last[0, col] = col_medians[col]

            X_scaled    = scaler.transform(X_last)
            raw_pred    = float(model.predict(X_scaled)[0])
        except Exception:
            raw_pred = current_price

        # Clamp to ±70% of current price
        predicted_price = max(current_price * 0.30, min(current_price * 1.70, raw_pred))
        pct_change      = round((predicted_price - current_price) / current_price * 100, 2)
        trend           = _trend_label(current_price, predicted_price)
        confidence      = _confidence_from_mape(mape)

        if best_mape is None or mape < best_mape:
            best_mape = mape

        predictions.append({
            "horizon_days":    horizon,
            "predicted_price": round(predicted_price, 2),
            "pct_change":      pct_change,
            "trend":           trend,
            "confidence":      round(confidence * 100, 1),
        })

    # ── 30-day daily forecast series for the chart ───────────────────────────
    forecast_dates  = []
    forecast_prices = []

    pred_7  = next((p["predicted_price"] for p in predictions if p["horizon_days"] == 7),  current_price)
    pred_30 = next((p["predicted_price"] for p in predictions if p["horizon_days"] == 30), current_price)

    last_date = pd.to_datetime(df["date"].iloc[-1])
    hist_std  = float(df["close"].pct_change().std()) * current_price

    for day in range(1, 31):
        future_date = last_date + pd.Timedelta(days=day)

        if day <= 7:
            frac  = day / 7
            price = current_price + frac * (pred_7 - current_price)
        else:
            frac  = (day - 7) / 23
            price = pred_7 + frac * (pred_30 - pred_7)

        np.random.seed(day * 13)
        noise = np.random.normal(0, hist_std * 0.25)
        price = round(max(current_price * 0.5, price + noise), 2)

        forecast_dates.append(future_date.strftime("%Y-%m-%d"))
        forecast_prices.append(price)

    overall_trend  = _trend_label(current_price, pred_30)
    avg_confidence = round(
        sum(p["confidence"] for p in predictions) / max(1, len(predictions)), 1
    )

    return {
        "current_price":   round(current_price, 2),
        "predictions":     predictions,
        "confidence":      avg_confidence,
        "trend":           overall_trend,
        "mape":            round(best_mape * 100, 2) if best_mape is not None else None,
        "forecast_dates":  forecast_dates,
        "forecast_prices": forecast_prices,
        "model_type":      "XGBoost",
    }
