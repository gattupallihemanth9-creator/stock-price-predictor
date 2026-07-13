"""
app.py
Flask application entry point.
Run with:  python app.py
Visit:     http://localhost:5000
"""

import os
import uuid
import math
import json
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from flask import (
    Flask, render_template, request, jsonify,
    session, redirect, url_for
)
from flask.json.provider import DefaultJSONProvider
from sqlalchemy.exc import IntegrityError

from models import db, Stock, PriceHistory, Prediction, Watchlist
from analysis import build_indicators
from predictor import predict
from recommender import get_recommendations, DEFAULT_POOL
from nse import get_live_quote, get_nifty50_quotes, search_nse, NSE_STOCKS

# ---------------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------------

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# On Render (and most cloud platforms) only /tmp is writable at runtime
DB_DIR = os.environ.get("DB_DIR", BASE_DIR)
DB_PATH = os.path.join(DB_DIR, "database.db")


class SafeJSONProvider(DefaultJSONProvider):
    """Replace NaN / Inf with None before serialising to JSON."""

    def dumps(self, obj, **kwargs):
        return json.dumps(obj, default=self._default, **kwargs)

    @staticmethod
    def _default(o):
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            return None
        if isinstance(o, np.floating):
            v = float(o)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

app = Flask(__name__)
app.json_provider_class = SafeJSONProvider
app.json = SafeJSONProvider(app)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "stock-predictor-secret-2024")
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_PATH}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_session_id() -> str:
    """Return (or create) a persistent session ID for the current browser session."""
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return session["session_id"]


def fetch_and_cache(ticker: str) -> tuple[pd.DataFrame | None, dict | None]:
    """
    Download historical data from yfinance, upsert into SQLite,
    and return (dataframe, stock_info_dict).
    Returns (None, None) on failure.
    """
    ticker = ticker.upper().strip()

    try:
        info = yf.Ticker(ticker).info
        if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
            # Try a quick price check
            test = yf.download(ticker, period="5d", interval="1d", progress=False)
            if test.empty:
                return None, None

        raw = yf.download(ticker, period="max", interval="1d", progress=False, auto_adjust=True)
        if raw.empty or len(raw) < 30:
            return None, None

        raw = raw.reset_index()

        # Flatten multi-level columns that yfinance sometimes returns
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [col[0].lower() for col in raw.columns]
        else:
            raw.columns = [c.lower() for c in raw.columns]

        raw = raw.rename(columns={"adj close": "close", "price": "close"})
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in raw.columns:
                raw[col] = 0

        raw["date"] = pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d")
        df = raw[["date", "open", "high", "low", "close", "volume"]].dropna()
        df = df[df["close"] > 0]

        # Upsert stock record
        stock = Stock.query.filter_by(ticker=ticker).first()
        if not stock:
            stock = Stock(ticker=ticker)
            db.session.add(stock)

        stock.company_name = info.get("longName") or info.get("shortName") or ticker
        stock.sector = info.get("sector", "Unknown")
        stock.market_cap = info.get("marketCap")
        stock.last_updated = datetime.utcnow()

        try:
            db.session.commit()
        except Exception:
            db.session.rollback()
            stock = Stock.query.filter_by(ticker=ticker).first()

        # Upsert price rows (only new dates)
        existing_dates = {
            r.date.isoformat()
            for r in PriceHistory.query.filter_by(stock_id=stock.id).all()
        }

        new_rows = []
        for _, row in df.iterrows():
            if row["date"] not in existing_dates:
                new_rows.append(PriceHistory(
                    stock_id=stock.id,
                    date=datetime.strptime(row["date"], "%Y-%m-%d").date(),
                    open_price=float(row["open"]),
                    high_price=float(row["high"]),
                    low_price=float(row["low"]),
                    close_price=float(row["close"]),
                    volume=int(row["volume"]),
                ))

        if new_rows:
            db.session.bulk_save_objects(new_rows)
            db.session.commit()

        # Build stock_info dict (use _safe_float to guard against NaN from yfinance)
        _cp = info.get("currentPrice") or info.get("regularMarketPrice") or float(df["close"].iloc[-1])
        current_price = _safe_float(_cp) or float(df["close"].iloc[-1])
        stock_info = {
            "ticker": ticker,
            "company_name": stock.company_name,
            "sector": stock.sector,
            "market_cap": _safe_float(info.get("marketCap")),
            "current_price": round(current_price, 2),
            "fifty_two_week_high": _safe_float(info.get("fiftyTwoWeekHigh")),
            "fifty_two_week_low":  _safe_float(info.get("fiftyTwoWeekLow")),
            "pe_ratio":       _safe_float(info.get("trailingPE")),
            "dividend_yield": _safe_float(info.get("dividendYield")),
            "beta":           _safe_float(info.get("beta")),
        }
        return df, stock_info

    except Exception as e:
        app.logger.error(f"fetch_and_cache error for {ticker}: {e}")
        return None, None


def _safe_float(v):
    """Return v as a float, or None if NaN/Inf/None."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _stale(ticker: str, hours: int = 12) -> bool:
    """Return True if the cached data is older than `hours` hours."""
    stock = Stock.query.filter_by(ticker=ticker).first()
    if not stock or not stock.last_updated:
        return True
    return datetime.utcnow() - stock.last_updated > timedelta(hours=hours)


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/stock/<ticker>")
def stock_page(ticker):
    return render_template("stock.html", ticker=ticker.upper())


@app.route("/watchlist")
def watchlist_page():
    sid = get_session_id()
    items = Watchlist.query.filter_by(session_id=sid).order_by(Watchlist.added_at.desc()).all()
    tickers = [item.ticker for item in items]
    return render_template("watchlist.html", tickers=tickers)


# ---------------------------------------------------------------------------
# API – stock data
# ---------------------------------------------------------------------------
# In-memory prediction cache to avoid recomputing on every request
# { ticker: {"result": dict, "computed_at": datetime} }
# ---------------------------------------------------------------------------
_PRED_CACHE: dict = {}


@app.route("/api/stock/<ticker>")
def api_stock(ticker):
    """
    Returns stock analysis JSON: { stock_info, indicators, prediction }
    Prediction is served from cache if available, otherwise returns a
    'computing' status so the frontend can poll /api/stock/<ticker>/predict.
    """
    ticker = ticker.upper().strip()

    if _stale(ticker):
        df, stock_info = fetch_and_cache(ticker)
        if df is None:
            return jsonify({"error": f"Could not find data for ticker '{ticker}'. Please check the symbol."}), 404
    else:
        stock = Stock.query.filter_by(ticker=ticker).first()
        rows = (PriceHistory.query
                .filter_by(stock_id=stock.id)
                .order_by(PriceHistory.date.asc())
                .all())

        records = [r.to_dict() for r in rows]
        df = pd.DataFrame(records)
        df = df.rename(columns={
            "open": "open", "high": "high", "low": "low",
            "close": "close", "volume": "volume"
        })

        current_price = float(df["close"].iloc[-1])
        stock_info = {
            "ticker": ticker,
            "company_name": stock.company_name,
            "sector": stock.sector,
            "market_cap": stock.market_cap,
            "current_price": round(current_price, 2),
        }

    # Run technical analysis (fast — always included)
    try:
        indicators = build_indicators(df)
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

    # Check prediction cache first
    cached = _PRED_CACHE.get(ticker)
    if cached and (datetime.utcnow() - cached["computed_at"]).total_seconds() < 43200:
        prediction = cached["result"]
    else:
        # Run prediction synchronously — XGBoost is fast enough (2-3s)
        try:
            prediction = predict(df, ticker=ticker)
            _PRED_CACHE[ticker] = {
                "result":      prediction,
                "computed_at": datetime.utcnow(),
            }
        except Exception as e:
            prediction = {
                "error":       str(e),
                "predictions": [],
                "model_type":  "none",
            }

    # Persist predictions to DB
    if prediction.get("predictions"):
        stock = Stock.query.filter_by(ticker=ticker).first()
        if stock:
            for p in prediction["predictions"]:
                pred_obj = Prediction(
                    stock_id=stock.id,
                    horizon_days=p["horizon_days"],
                    predicted_price=p["predicted_price"],
                    confidence=p["confidence"] / 100,
                    trend=p["trend"],
                )
                db.session.add(pred_obj)
            try:
                db.session.commit()
            except Exception:
                db.session.rollback()

    return jsonify({
        "stock_info": stock_info,
        "indicators": indicators,
        "prediction": prediction,
    })


@app.route("/api/stock/<ticker>/history")
def api_history(ticker):
    """Returns last N days of OHLCV data. Default N=365."""
    ticker = ticker.upper().strip()
    days = request.args.get("days", 365, type=int)

    stock = Stock.query.filter_by(ticker=ticker).first()
    if not stock:
        df, _ = fetch_and_cache(ticker)
        if df is None:
            return jsonify({"error": "Ticker not found"}), 404
        stock = Stock.query.filter_by(ticker=ticker).first()

    cutoff = (datetime.utcnow() - timedelta(days=days)).date()
    rows = (PriceHistory.query
            .filter(PriceHistory.stock_id == stock.id, PriceHistory.date >= cutoff)
            .order_by(PriceHistory.date.asc())
            .all())

    return jsonify([r.to_dict() for r in rows])


# ---------------------------------------------------------------------------
# API – recommendations
# ---------------------------------------------------------------------------

@app.route("/api/recommendations")
def api_recommendations():
    """
    Returns top 5 recommended stocks.
    Query params:
      pool  – comma-separated list of tickers (optional, defaults to built-in pool)
      top_n – how many results to return (default 5)
    """
    pool_param = request.args.get("pool")
    top_n = request.args.get("top_n", 5, type=int)

    pool = [t.strip().upper() for t in pool_param.split(",")] if pool_param else None

    try:
        recs = get_recommendations(pool=pool, top_n=top_n)
        return jsonify({"recommendations": recs})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# API – watchlist
# ---------------------------------------------------------------------------

@app.route("/api/watchlist", methods=["GET"])
def api_watchlist_get():
    sid = get_session_id()
    items = Watchlist.query.filter_by(session_id=sid).order_by(Watchlist.added_at.desc()).all()
    return jsonify({"watchlist": [i.to_dict() for i in items]})


@app.route("/api/watchlist/add", methods=["POST"])
def api_watchlist_add():
    sid = get_session_id()
    data = request.get_json(force=True)
    ticker = (data.get("ticker") or "").upper().strip()

    if not ticker:
        return jsonify({"error": "Ticker is required"}), 400

    entry = Watchlist(session_id=sid, ticker=ticker)
    db.session.add(entry)
    try:
        db.session.commit()
        return jsonify({"message": f"{ticker} added to watchlist"})
    except IntegrityError:
        db.session.rollback()
        return jsonify({"message": f"{ticker} is already in your watchlist"})


@app.route("/api/watchlist/remove", methods=["DELETE"])
def api_watchlist_remove():
    sid = get_session_id()
    data = request.get_json(force=True)
    ticker = (data.get("ticker") or "").upper().strip()

    if not ticker:
        return jsonify({"error": "Ticker is required"}), 400

    entry = Watchlist.query.filter_by(session_id=sid, ticker=ticker).first()
    if entry:
        db.session.delete(entry)
        db.session.commit()
        return jsonify({"message": f"{ticker} removed from watchlist"})
    return jsonify({"error": "Ticker not found in watchlist"}), 404


# ---------------------------------------------------------------------------
# API – search suggestions
# ---------------------------------------------------------------------------

POPULAR_TICKERS = [
    {"ticker": t, "name": n, "exchange": e} for t, n, e in [
        # US stocks
        ("AAPL",  "Apple Inc.",              "NASDAQ"),
        ("MSFT",  "Microsoft Corporation",   "NASDAQ"),
        ("GOOGL", "Alphabet Inc.",           "NASDAQ"),
        ("AMZN",  "Amazon.com Inc.",         "NASDAQ"),
        ("NVDA",  "NVIDIA Corporation",      "NASDAQ"),
        ("TSLA",  "Tesla Inc.",              "NASDAQ"),
        ("META",  "Meta Platforms Inc.",     "NASDAQ"),
        ("JPM",   "JPMorgan Chase",          "NYSE"),
        ("V",     "Visa Inc.",               "NYSE"),
        # NSE India stocks
        ("RELIANCE.NS",   "Reliance Industries",           "NSE"),
        ("TCS.NS",        "Tata Consultancy Services",     "NSE"),
        ("HDFCBANK.NS",   "HDFC Bank",                    "NSE"),
        ("INFY.NS",       "Infosys",                       "NSE"),
        ("ICICIBANK.NS",  "ICICI Bank",                    "NSE"),
        ("SBIN.NS",       "State Bank of India",           "NSE"),
        ("BHARTIARTL.NS", "Bharti Airtel",                 "NSE"),
        ("WIPRO.NS",      "Wipro",                         "NSE"),
        ("KOTAKBANK.NS",  "Kotak Mahindra Bank",           "NSE"),
        ("LT.NS",         "Larsen & Toubro",               "NSE"),
        ("AXISBANK.NS",   "Axis Bank",                     "NSE"),
        ("MARUTI.NS",     "Maruti Suzuki",                 "NSE"),
        ("SUNPHARMA.NS",  "Sun Pharmaceutical",            "NSE"),
        ("TITAN.NS",      "Titan Company",                 "NSE"),
        ("BAJFINANCE.NS", "Bajaj Finance",                 "NSE"),
        ("HCLTECH.NS",    "HCL Technologies",              "NSE"),
        ("TATAMOTORS.NS", "Tata Motors",                   "NSE"),
        ("ADANIENT.NS",   "Adani Enterprises",             "NSE"),
        ("ITC.NS",        "ITC",                           "NSE"),
        ("M&M.NS",        "Mahindra & Mahindra",           "NSE"),
    ]
]


@app.route("/api/search")
def api_search():
    q = request.args.get("q", "").upper().strip()
    if not q:
        return jsonify({"results": []})

    # Match from popular tickers (US + NSE)
    matches = [t for t in POPULAR_TICKERS if q in t["ticker"] or q in t["name"].upper()]

    # Also search full NSE directory for any extra hits
    nse_hits = search_nse(q)
    existing = {m["ticker"] for m in matches}
    for hit in nse_hits:
        if hit["ticker"] not in existing:
            matches.append(hit)

    return jsonify({"results": matches[:12]})


# ---------------------------------------------------------------------------
# API – NSE India live prices
# ---------------------------------------------------------------------------

@app.route("/api/nse/quote/<symbol>")
def api_nse_quote(symbol):
    """
    Returns live NSE quote for a symbol.
    Accepts with or without .NS suffix.
    e.g. /api/nse/quote/RELIANCE  or  /api/nse/quote/RELIANCE.NS
    """
    sym = symbol.upper().strip()
    if not sym.endswith(".NS"):
        sym += ".NS"

    quote = get_live_quote(sym)
    if quote is None:
        return jsonify({"error": f"Could not fetch live quote for {sym}"}), 404

    return jsonify(quote)


@app.route("/api/nse/market")
def api_nse_market():
    """
    Returns live quotes for top NSE stocks (Nifty 50 subset).
    Query param: symbols=RELIANCE,TCS,INFY  (optional, comma-separated, no .NS needed)
    """
    symbols_param = request.args.get("symbols")
    if symbols_param:
        symbols = [
            s.strip().upper() + (".NS" if not s.strip().upper().endswith(".NS") else "")
            for s in symbols_param.split(",")
        ]
    else:
        symbols = None   # defaults to top 20 Nifty in nse.py

    try:
        quotes = get_nifty50_quotes(symbols)
        return jsonify({
            "quotes": quotes,
            "count": len(quotes),
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/nse/search")
def api_nse_search():
    """Search NSE stock directory. ?q=RELIANCE"""
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"results": []})
    results = search_nse(q)
    return jsonify({"results": results})


# ---------------------------------------------------------------------------
# Bootstrap DB and run
# ---------------------------------------------------------------------------

with app.app_context():
    db.create_all()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
