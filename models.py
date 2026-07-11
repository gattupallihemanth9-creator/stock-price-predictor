from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()


class Stock(db.Model):
    """Stores basic stock metadata."""
    __tablename__ = "stocks"

    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(20), unique=True, nullable=False)
    company_name = db.Column(db.String(200))
    sector = db.Column(db.String(100))
    market_cap = db.Column(db.Float)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)

    price_history = db.relationship("PriceHistory", backref="stock", lazy=True, cascade="all, delete-orphan")
    predictions = db.relationship("Prediction", backref="stock", lazy=True, cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "ticker": self.ticker,
            "company_name": self.company_name,
            "sector": self.sector,
            "market_cap": self.market_cap,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }


class PriceHistory(db.Model):
    """Stores daily OHLCV price data for each stock."""
    __tablename__ = "price_history"

    id = db.Column(db.Integer, primary_key=True)
    stock_id = db.Column(db.Integer, db.ForeignKey("stocks.id"), nullable=False)
    date = db.Column(db.Date, nullable=False)
    open_price = db.Column(db.Float)
    high_price = db.Column(db.Float)
    low_price = db.Column(db.Float)
    close_price = db.Column(db.Float)
    volume = db.Column(db.BigInteger)

    __table_args__ = (db.UniqueConstraint("stock_id", "date", name="uq_stock_date"),)

    def to_dict(self):
        return {
            "date": self.date.isoformat(),
            "open": self.open_price,
            "high": self.high_price,
            "low": self.low_price,
            "close": self.close_price,
            "volume": self.volume,
        }


class Prediction(db.Model):
    """Stores ML prediction results for a stock."""
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    stock_id = db.Column(db.Integer, db.ForeignKey("stocks.id"), nullable=False)
    predicted_at = db.Column(db.DateTime, default=datetime.utcnow)
    horizon_days = db.Column(db.Integer, nullable=False)   # 7, 30, or 90
    predicted_price = db.Column(db.Float)
    confidence = db.Column(db.Float)                       # 0.0 – 1.0
    trend = db.Column(db.String(20))                       # Bullish / Bearish / Neutral

    def to_dict(self):
        return {
            "predicted_at": self.predicted_at.isoformat(),
            "horizon_days": self.horizon_days,
            "predicted_price": round(self.predicted_price, 2) if self.predicted_price else None,
            "confidence": round(self.confidence * 100, 1) if self.confidence else None,
            "trend": self.trend,
        }


class Watchlist(db.Model):
    """User watchlist – stored by session ID (no login required)."""
    __tablename__ = "watchlist"

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(100), nullable=False)
    ticker = db.Column(db.String(20), nullable=False)
    added_at = db.Column(db.DateTime, default=datetime.utcnow)

    __table_args__ = (db.UniqueConstraint("session_id", "ticker", name="uq_session_ticker"),)

    def to_dict(self):
        return {
            "ticker": self.ticker,
            "added_at": self.added_at.isoformat(),
        }
