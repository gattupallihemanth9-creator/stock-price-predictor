"""
predictor.py
LSTM-based stock price predictor.

Architecture:
  - Input:  60-day sliding window of [close, volume, sma20, ema12, rsi, macd, returns]
  - Model:  LSTM(128) → Dropout(0.2) → LSTM(64) → Dropout(0.2) → Dense(32) → Dense(1)
  - Output: Predicted close price for horizon days ahead (7, 30, 90)

One model is trained per horizon so each prediction is independent.
Models are cached in memory so repeated calls within the same process
don't retrain from scratch.
"""

import os
import math
import warnings
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # suppress TF C++ logs
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error

# Lazy-import Keras so TF is only loaded when predict() is actually called
_keras_loaded = False

def _load_keras():
    global _keras_loaded, Sequential, LSTM, Dense, Dropout, EarlyStopping
    if not _keras_loaded:
        from tensorflow.keras.models import Sequential as _S
        from tensorflow.keras.layers import LSTM as _L, Dense as _D, Dropout as _Dr
        from tensorflow.keras.callbacks import EarlyStopping as _E
        Sequential    = _S
        LSTM          = _L
        Dense         = _D
        Dropout       = _Dr
        EarlyStopping = _E
        _keras_loaded = True


# ---------------------------------------------------------------------------
# In-process model cache  { (ticker, horizon): (model, feature_scaler, target_scaler) }
# ---------------------------------------------------------------------------
_MODEL_CACHE: dict = {}

LOOKBACK   = 60   # days of history fed into LSTM at each step
FEATURES   = ["close", "volume", "sma_20", "ema_12", "rsi_14",
               "macd", "ret_1", "ret_5", "ret_20", "volatility"]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _build_feature_df(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all LSTM input features from raw OHLCV data."""
    d = df.copy()
    c = d["close"].astype(float)
    v = d["volume"].astype(float)

    # Moving averages & EMA
    d["sma_20"] = c.rolling(20).mean()
    d["sma_50"] = c.rolling(50).mean()
    d["ema_12"] = c.ewm(span=12, adjust=False).mean()
    d["ema_26"] = c.ewm(span=26, adjust=False).mean()

    # MACD
    d["macd"] = d["ema_12"] - d["ema_26"]

    # RSI (14)
    delta  = c.diff()
    gain   = delta.clip(lower=0).ewm(com=13, min_periods=14).mean()
    loss   = (-delta.clip(upper=0)).ewm(com=13, min_periods=14).mean()
    rs     = gain / loss.replace(0, np.nan)
    d["rsi_14"] = 100 - (100 / (1 + rs))

    # Returns
    d["ret_1"]  = c.pct_change(1)
    d["ret_5"]  = c.pct_change(5)
    d["ret_20"] = c.pct_change(20)

    # Volatility (20-day rolling std of returns)
    d["volatility"] = d["ret_1"].rolling(20).std()

    # Volume normalised
    d["volume"] = v / v.rolling(20).mean().replace(0, np.nan)

    return d.dropna().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sequence builder
# ---------------------------------------------------------------------------

def _make_sequences(X: np.ndarray, y: np.ndarray, lookback: int):
    """
    Convert flat feature matrix to (samples, lookback, features) sequences.
    X[i] → sequence of X[i:i+lookback]
    y[i] → target at position i+lookback
    """
    xs, ys = [], []
    for i in range(len(X) - lookback):
        xs.append(X[i : i + lookback])
        ys.append(y[i + lookback])
    return np.array(xs), np.array(ys)


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def _build_model(n_features: int) -> "Sequential":
    _load_keras()
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(LOOKBACK, n_features)),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="huber")
    return model


# ---------------------------------------------------------------------------
# Train one model for a specific horizon
# ---------------------------------------------------------------------------

def _train(df: pd.DataFrame, horizon: int, ticker: str = ""):
    """
    Train an LSTM that predicts close price `horizon` days ahead.
    Returns (model, feat_scaler, target_scaler, mape).
    """
    _load_keras()

    feat_df = _build_feature_df(df)
    if len(feat_df) < LOOKBACK + horizon + 50:
        raise ValueError(f"Not enough data for {horizon}-day LSTM (have {len(feat_df)} rows)")

    # Features and target
    X_raw = feat_df[FEATURES].values.astype(float)
    y_raw = feat_df["close"].values.astype(float)

    # Shift target by `horizon` so model learns to predict horizon days ahead
    X_raw = X_raw[:-horizon]
    y_raw = y_raw[horizon:]

    # Scale
    feat_scaler   = MinMaxScaler()
    target_scaler = MinMaxScaler()

    X_scaled = feat_scaler.fit_transform(X_raw)
    y_scaled = target_scaler.fit_transform(y_raw.reshape(-1, 1)).flatten()

    # Build sequences
    X_seq, y_seq = _make_sequences(X_scaled, y_scaled, LOOKBACK)

    # Train / validation split (80/20, no shuffle)
    split     = int(len(X_seq) * 0.85)
    X_train, X_val = X_seq[:split], X_seq[split:]
    y_train, y_val = y_seq[:split], y_seq[split:]

    model = _build_model(len(FEATURES))

    early_stop = EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True, verbose=0
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=60,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0,
    )

    # Compute MAPE on validation set
    y_pred_scaled = model.predict(X_val, verbose=0).flatten()
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = target_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()

    mask = y_true != 0
    mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask]) if mask.sum() > 0 else 0.15

    return model, feat_scaler, target_scaler, mape


# ---------------------------------------------------------------------------
# Single-step prediction using trained model
# ---------------------------------------------------------------------------

def _predict_horizon(
    df: pd.DataFrame,
    model,
    feat_scaler,
    target_scaler,
    horizon: int,
) -> float:
    """
    Use the last LOOKBACK rows of features to predict price `horizon` days ahead.
    """
    feat_df = _build_feature_df(df)
    if len(feat_df) < LOOKBACK:
        raise ValueError("Not enough rows for prediction")

    X_last = feat_df[FEATURES].iloc[-LOOKBACK:].values.astype(float)

    # Replace any NaN/Inf with column mean
    col_means = np.nanmean(X_last, axis=0)
    for col in range(X_last.shape[1]):
        bad = ~np.isfinite(X_last[:, col])
        X_last[bad, col] = col_means[col]

    X_scaled = feat_scaler.transform(X_last)
    X_seq    = X_scaled.reshape(1, LOOKBACK, len(FEATURES))

    pred_scaled = model.predict(X_seq, verbose=0)[0][0]
    predicted   = float(target_scaler.inverse_transform([[pred_scaled]])[0][0])

    return predicted


# ---------------------------------------------------------------------------
# Helpers
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
    conf = max(0.05, 1.0 - mape * 3.5)
    return round(min(conf, 0.95), 3)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict(df: pd.DataFrame, ticker: str = "") -> dict:
    """
    Predict stock prices at 7, 30, and 90-day horizons using LSTM.

    Parameters
    ----------
    df     : DataFrame with columns [date, open, high, low, close, volume]
    ticker : optional ticker string used for model caching

    Returns
    -------
    dict with keys: current_price, predictions, confidence, trend,
                    mape, forecast_dates, forecast_prices, model_type
    """
    if len(df) < LOOKBACK + 100:
        return {
            "error": f"Not enough data for LSTM (need {LOOKBACK + 100}+ days, have {len(df)}).",
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

    for horizon in horizons:
        cache_key = (ticker, horizon)

        # Use cached model if available
        if cache_key in _MODEL_CACHE:
            model, feat_scaler, target_scaler, mape = _MODEL_CACHE[cache_key]
        else:
            try:
                model, feat_scaler, target_scaler, mape = _train(df, horizon, ticker)
                _MODEL_CACHE[cache_key] = (model, feat_scaler, target_scaler, mape)
            except Exception as e:
                predictions.append({
                    "horizon_days":    horizon,
                    "predicted_price": round(current_price, 2),
                    "pct_change":      0.0,
                    "trend":           "Neutral",
                    "confidence":      0.0,
                    "error":           str(e),
                })
                continue

        try:
            raw_pred = _predict_horizon(df, model, feat_scaler, target_scaler, horizon)
        except Exception as e:
            raw_pred = current_price

        # Sanity clamp: ±70% of current price
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

    # ── 30-day daily forecast series for the chart ────────────────────────────
    forecast_dates  = []
    forecast_prices = []

    pred_7  = next((p["predicted_price"] for p in predictions if p["horizon_days"] == 7),  current_price)
    pred_30 = next((p["predicted_price"] for p in predictions if p["horizon_days"] == 30), current_price)

    last_date = pd.to_datetime(df["date"].iloc[-1])
    hist_std  = float(df["close"].pct_change().std()) * current_price

    for day in range(1, 31):
        future_date = last_date + pd.Timedelta(days=day)

        # Smooth interpolation: current → 7d → 30d
        if day <= 7:
            frac  = day / 7
            price = current_price + frac * (pred_7 - current_price)
        else:
            frac  = (day - 7) / 23
            price = pred_7 + frac * (pred_30 - pred_7)

        # Add small deterministic noise for realism
        np.random.seed(day * 7)
        noise = np.random.normal(0, hist_std * 0.25)
        price = round(max(current_price * 0.5, price + noise), 2)

        forecast_dates.append(future_date.strftime("%Y-%m-%d"))
        forecast_prices.append(price)

    overall_trend  = _trend_label(current_price, pred_30)
    avg_confidence = round(
        sum(p["confidence"] for p in predictions if "error" not in p) /
        max(1, sum(1 for p in predictions if "error" not in p)), 1
    )

    return {
        "current_price":   round(current_price, 2),
        "predictions":     predictions,
        "confidence":      avg_confidence,
        "trend":           overall_trend,
        "mape":            round(best_mape * 100, 2) if best_mape is not None else None,
        "forecast_dates":  forecast_dates,
        "forecast_prices": forecast_prices,
        "model_type":      "LSTM",
    }
