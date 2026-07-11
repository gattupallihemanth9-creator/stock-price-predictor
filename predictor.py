"""
predictor.py
Uses Linear Regression on engineered features to predict future stock prices.
Horizons: 7, 30, 90 days.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag and rolling features to the close-price series."""
    df = df.copy()
    close = df["close"]

    # Lag features
    for lag in [1, 3, 5, 10, 20]:
        df[f"lag_{lag}"] = close.shift(lag)

    # Rolling statistics
    for w in [5, 10, 20, 50]:
        df[f"roll_mean_{w}"] = close.rolling(w).mean()
        df[f"roll_std_{w}"] = close.rolling(w).std()

    # Momentum
    df["momentum_5"] = close - close.shift(5)
    df["momentum_10"] = close - close.shift(10)

    # Day-of-year as a cyclic feature
    df["day_sin"] = np.sin(2 * np.pi * pd.to_datetime(df["date"]).dt.dayofyear / 365)
    df["day_cos"] = np.cos(2 * np.pi * pd.to_datetime(df["date"]).dt.dayofyear / 365)

    return df.dropna()


def _train_model(df: pd.DataFrame):
    """Train a Linear Regression model. Returns (model, scaler, mape)."""
    df = _engineer_features(df)
    feature_cols = [c for c in df.columns if c not in ("date", "open", "high", "low", "close", "volume")]

    X = df[feature_cols].values
    y = df["close"].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    if len(X_scaled) < 30:
        # Not enough data – fall back to a trivial model
        model = LinearRegression()
        model.fit(X_scaled, y)
        return model, scaler, feature_cols, None

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, shuffle=False
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    return model, scaler, feature_cols, mape


def _confidence_from_mape(mape) -> float:
    """Convert MAPE to a 0-1 confidence score."""
    if mape is None:
        return 0.5
    # MAPE of 0% → confidence 1.0, MAPE of 20%+ → confidence ~0.0
    confidence = max(0.0, 1.0 - mape * 5)
    return round(min(confidence, 1.0), 3)


def _trend_label(current_price: float, predicted_price: float) -> str:
    pct = (predicted_price - current_price) / current_price * 100
    if pct > 2:
        return "Bullish"
    elif pct < -2:
        return "Bearish"
    return "Neutral"


def predict(df: pd.DataFrame) -> dict:
    """
    Main entry point.

    Parameters
    ----------
    df : DataFrame with columns [date, open, high, low, close, volume]

    Returns
    -------
    dict with keys: predictions (list), confidence, trend, forecast_dates, forecast_prices
    """
    if len(df) < 60:
        return {
            "error": "Not enough historical data for prediction (need at least 60 days).",
            "predictions": [],
        }

    df = df.sort_values("date").reset_index(drop=True)
    current_price = float(df["close"].iloc[-1])

    model, scaler, feature_cols, mape = _train_model(df)
    confidence = _confidence_from_mape(mape)

    horizons = [7, 30, 90]
    predictions = []

    # Iterative multi-step forecast
    future_df = df.copy()

    for horizon in horizons:
        temp_df = future_df.copy()

        # Simulate 'horizon' days forward by rolling the last row
        for _ in range(horizon):
            last_close = temp_df["close"].iloc[-1]
            last_date = pd.to_datetime(temp_df["date"].iloc[-1]) + pd.Timedelta(days=1)

            new_row = {
                "date": last_date.strftime("%Y-%m-%d"),
                "open": last_close,
                "high": last_close,
                "low": last_close,
                "close": last_close,
                "volume": int(temp_df["volume"].mean()),
            }
            temp_df = pd.concat([temp_df, pd.DataFrame([new_row])], ignore_index=True)

            # Recompute features and predict only the latest row
            engineered = _engineer_features(temp_df)
            if engineered.empty:
                continue

            X_last = engineered[feature_cols].iloc[-1:].values
            X_last_scaled = scaler.transform(X_last)
            predicted = float(model.predict(X_last_scaled)[0])

            # Update the last close with the prediction
            temp_df.at[len(temp_df) - 1, "close"] = predicted

        predicted_price = float(temp_df["close"].iloc[-1])
        trend = _trend_label(current_price, predicted_price)
        pct_change = round((predicted_price - current_price) / current_price * 100, 2)

        predictions.append({
            "horizon_days": horizon,
            "predicted_price": round(predicted_price, 2),
            "pct_change": pct_change,
            "trend": trend,
            "confidence": round(confidence * 100, 1),
        })

    # Build a 30-day daily forecast series for the chart
    forecast_dates = []
    forecast_prices = []
    rolling_df = df.copy()

    for i in range(30):
        last_close = rolling_df["close"].iloc[-1]
        last_date = pd.to_datetime(rolling_df["date"].iloc[-1]) + pd.Timedelta(days=1)

        new_row = {
            "date": last_date.strftime("%Y-%m-%d"),
            "open": last_close,
            "high": last_close,
            "low": last_close,
            "close": last_close,
            "volume": int(rolling_df["volume"].mean()),
        }
        rolling_df = pd.concat([rolling_df, pd.DataFrame([new_row])], ignore_index=True)

        engineered = _engineer_features(rolling_df)
        if not engineered.empty:
            X_last = engineered[feature_cols].iloc[-1:].values
            X_last_scaled = scaler.transform(X_last)
            predicted = float(model.predict(X_last_scaled)[0])
            rolling_df.at[len(rolling_df) - 1, "close"] = predicted

        forecast_dates.append(last_date.strftime("%Y-%m-%d"))
        forecast_prices.append(round(float(rolling_df["close"].iloc[-1]), 2))

    overall_trend = _trend_label(current_price, forecast_prices[-1] if forecast_prices else current_price)

    return {
        "current_price": round(current_price, 2),
        "predictions": predictions,
        "confidence": round(confidence * 100, 1),
        "trend": overall_trend,
        "mape": round(mape * 100, 2) if mape is not None else None,
        "forecast_dates": forecast_dates,
        "forecast_prices": forecast_prices,
    }
