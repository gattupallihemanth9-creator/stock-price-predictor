"""
nse.py
NSE India live price and market data via yfinance (.NS suffix).

All NSE tickers must end with .NS  e.g. RELIANCE.NS, TCS.NS
Prices are in INR (₹).
"""

import math
import yfinance as yf
import pandas as pd
from datetime import datetime

# ---------------------------------------------------------------------------
# NSE stock directory – symbol → display name
# ---------------------------------------------------------------------------

NSE_STOCKS = {
    # Nifty 50 blue chips
    "RELIANCE.NS":    "Reliance Industries",
    "TCS.NS":         "Tata Consultancy Services",
    "HDFCBANK.NS":    "HDFC Bank",
    "INFY.NS":        "Infosys",
    "ICICIBANK.NS":   "ICICI Bank",
    "HINDUNILVR.NS":  "Hindustan Unilever",
    "SBIN.NS":        "State Bank of India",
    "BHARTIARTL.NS":  "Bharti Airtel",
    "KOTAKBANK.NS":   "Kotak Mahindra Bank",
    "LT.NS":          "Larsen & Toubro",
    "AXISBANK.NS":    "Axis Bank",
    "ASIANPAINT.NS":  "Asian Paints",
    "MARUTI.NS":      "Maruti Suzuki",
    "SUNPHARMA.NS":   "Sun Pharmaceutical",
    "TITAN.NS":       "Titan Company",
    "WIPRO.NS":       "Wipro",
    "BAJFINANCE.NS":  "Bajaj Finance",
    "HCLTECH.NS":     "HCL Technologies",
    "NESTLEIND.NS":   "Nestle India",
    "POWERGRID.NS":   "Power Grid Corp",
    "NTPC.NS":        "NTPC",
    "ONGC.NS":        "Oil & Natural Gas Corp",
    "JSWSTEEL.NS":    "JSW Steel",
    "TATAMOTORS.NS":  "Tata Motors",
    "ADANIENT.NS":    "Adani Enterprises",
    "ADANIPORTS.NS":  "Adani Ports",
    "COALINDIA.NS":   "Coal India",
    "DRREDDY.NS":     "Dr Reddy's Laboratories",
    "CIPLA.NS":       "Cipla",
    "EICHERMOT.NS":   "Eicher Motors",
    "BRITANNIA.NS":   "Britannia Industries",
    "DIVISLAB.NS":    "Divi's Laboratories",
    "BAJAJ-AUTO.NS":  "Bajaj Auto",
    "HEROMOTOCO.NS":  "Hero MotoCorp",
    "GRASIM.NS":      "Grasim Industries",
    "ULTRACEMCO.NS":  "UltraTech Cement",
    "INDUSINDBK.NS":  "IndusInd Bank",
    "M&M.NS":         "Mahindra & Mahindra",
    "TATASTEEL.NS":   "Tata Steel",
    "APOLLOHOSP.NS":  "Apollo Hospitals",
    "TECHM.NS":       "Tech Mahindra",
    "HDFCLIFE.NS":    "HDFC Life Insurance",
    "SBILIFE.NS":     "SBI Life Insurance",
    "BPCL.NS":        "Bharat Petroleum",
    "HINDALCO.NS":    "Hindalco Industries",
    "UPL.NS":         "UPL",
    "TATACONSUM.NS":  "Tata Consumer Products",
    "BAJAJFINSV.NS":  "Bajaj Finserv",
    "ITC.NS":         "ITC",
    "SHREECEM.NS":    "Shree Cement",
}

# Nifty 50 default pool for recommendations
NSE_DEFAULT_POOL = list(NSE_STOCKS.keys())[:20]


def _safe(v):
    """Convert NaN/Inf/None to None."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 2)
    except (TypeError, ValueError):
        return None


def get_live_quote(symbol: str) -> dict | None:
    """
    Fetch live NSE price for a single symbol using yfinance fast_info.
    symbol should include .NS suffix, e.g. 'RELIANCE.NS'

    Returns dict with keys:
        ticker, company_name, last_price, prev_close, change, pct_change,
        day_high, day_low, volume, market_cap, currency, timestamp
    Returns None on failure.
    """
    symbol = symbol.upper().strip()
    if not symbol.endswith(".NS"):
        symbol += ".NS"

    try:
        t = yf.Ticker(symbol)
        fi = t.fast_info

        last_price  = _safe(fi.last_price)
        prev_close  = _safe(fi.previous_close)
        day_high    = _safe(fi.day_high)
        day_low     = _safe(fi.day_low)
        volume      = int(fi.last_volume) if fi.last_volume else None
        market_cap  = _safe(fi.market_cap)

        if last_price is None:
            return None

        change     = _safe(last_price - prev_close) if prev_close else None
        pct_change = _safe((change / prev_close * 100)) if (change is not None and prev_close) else None

        company_name = NSE_STOCKS.get(symbol) or symbol.replace(".NS", "")

        return {
            "ticker":       symbol,
            "symbol":       symbol.replace(".NS", ""),   # clean symbol for display
            "company_name": company_name,
            "last_price":   last_price,
            "prev_close":   prev_close,
            "change":       change,
            "pct_change":   pct_change,
            "day_high":     day_high,
            "day_low":      day_low,
            "volume":       volume,
            "market_cap":   market_cap,
            "currency":     "INR",
            "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    except Exception:
        return None


def get_nifty50_quotes(symbols: list[str] = None) -> list[dict]:
    """
    Fetch live quotes for a list of NSE symbols.
    Defaults to the top 20 Nifty 50 stocks.
    Returns list of quote dicts sorted by pct_change descending.
    """
    if symbols is None:
        symbols = NSE_DEFAULT_POOL

    results = []
    for sym in symbols:
        q = get_live_quote(sym)
        if q:
            results.append(q)

    results.sort(key=lambda x: (x.get("pct_change") or 0), reverse=True)
    return results


def search_nse(query: str) -> list[dict]:
    """
    Search NSE stocks by symbol or company name.
    Returns list of {ticker, symbol, name} dicts.
    """
    q = query.upper().strip()
    matches = []
    for ticker, name in NSE_STOCKS.items():
        symbol = ticker.replace(".NS", "")
        if q in symbol or q in name.upper():
            matches.append({
                "ticker": ticker,
                "symbol": symbol,
                "name":   name,
                "exchange": "NSE",
            })
    return matches[:10]
