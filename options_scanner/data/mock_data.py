"""
mock_data.py
------------
Realistic mock options data for development/testing.
Mirrors the exact structure that comes from Schwab's API,
so swapping to live data requires zero changes in the scanner logic.
"""

import random
from datetime import datetime, timedelta


def _expiry(days: int) -> str:
    return (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")


def get_mock_options_chain(symbol: str) -> dict:
    """
    Returns a mock options chain in Schwab API format for a given symbol.
    Prices and greeks are randomized within realistic ranges per symbol.
    """
    # Rough "current price" per symbol
    prices = {
        "AAPL": 221.50, "MSFT": 415.00, "NVDA": 138.00, "META": 632.00,
        "GOOGL": 192.00, "AMZN": 227.00, "TSLA": 295.00, "SPY": 590.00,
        "QQQ": 505.00, "GLD": 231.00, "AMD": 112.00, "CRM": 325.00,
        "NFLX": 1015.00, "UBER": 82.00, "COIN": 255.00,
    }
    underlying_price = prices.get(symbol, 100.00) * random.uniform(0.97, 1.03)

    call_map = {}
    put_map  = {}

    expirations = [14, 21, 28, 35, 42]

    for dte in expirations:
        exp_date = _expiry(dte)
        calls = []
        puts  = []

        for offset in [-0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15, 0.20]:
            strike = round(underlying_price * (1 + offset), 0)
            moneyness = offset  # negative = OTM call, positive = ITM call

            # Call
            call_delta  = max(0.05, min(0.95, 0.50 - moneyness * 2 + random.uniform(-0.05, 0.05)))
            call_iv     = random.uniform(0.25, 0.85)
            call_bid    = max(0.05, round(underlying_price * call_iv * (dte/365)**0.5 * call_delta * 0.9, 2))
            call_ask    = round(call_bid * random.uniform(1.02, 1.12), 2)
            call_oi     = random.randint(100, 80000)
            call_vol    = random.randint(10, call_oi // 2)

            calls.append({
                "strikePrice": strike,
                "expirationDate": exp_date,
                "daysToExpiration": dte,
                "bid": call_bid,
                "ask": call_ask,
                "mark": round((call_bid + call_ask) / 2, 2),
                "delta": round(call_delta, 3),
                "gamma": round(random.uniform(0.005, 0.05), 4),
                "theta": round(-random.uniform(0.01, 0.15), 4),
                "vega":  round(random.uniform(0.05, 0.35), 4),
                "volatility": round(call_iv * 100, 2),  # as percentage
                "openInterest": call_oi,
                "totalVolume": call_vol,
                "inTheMoney": moneyness < 0,
                "putCall": "CALL",
            })

            # Put (mirror delta)
            put_delta  = round(call_delta - 1, 3)
            put_iv     = call_iv * random.uniform(0.95, 1.15)
            put_bid    = max(0.05, round(underlying_price * put_iv * (dte/365)**0.5 * abs(put_delta) * 0.9, 2))
            put_ask    = round(put_bid * random.uniform(1.02, 1.12), 2)
            put_oi     = random.randint(100, 60000)
            put_vol    = random.randint(10, put_oi // 2)

            puts.append({
                "strikePrice": strike,
                "expirationDate": exp_date,
                "daysToExpiration": dte,
                "bid": put_bid,
                "ask": put_ask,
                "mark": round((put_bid + put_ask) / 2, 2),
                "delta": put_delta,
                "gamma": round(random.uniform(0.005, 0.05), 4),
                "theta": round(-random.uniform(0.01, 0.15), 4),
                "vega":  round(random.uniform(0.05, 0.35), 4),
                "volatility": round(put_iv * 100, 2),
                "openInterest": put_oi,
                "totalVolume": put_vol,
                "inTheMoney": moneyness > 0,
                "putCall": "PUT",
            })

        call_map[exp_date] = calls
        put_map[exp_date]  = puts

    return {
        "symbol": symbol,
        "underlyingPrice": round(underlying_price, 2),
        "callExpDateMap": call_map,
        "putExpDateMap":  put_map,
        "status": "SUCCESS",
    }


def get_mock_quote(symbol: str) -> dict:
    prices = {
        "AAPL": 221.50, "MSFT": 415.00, "NVDA": 138.00, "META": 632.00,
        "GOOGL": 192.00, "AMZN": 227.00, "TSLA": 295.00, "SPY": 590.00,
        "QQQ": 505.00, "GLD": 231.00, "AMD": 112.00, "CRM": 325.00,
        "NFLX": 1015.00, "UBER": 82.00, "COIN": 255.00,
    }
    price = prices.get(symbol, 100.00) * random.uniform(0.97, 1.03)
    return {
        "symbol": symbol,
        "lastPrice": round(price, 2),
        "bidPrice":  round(price * 0.999, 2),
        "askPrice":  round(price * 1.001, 2),
        "openPrice": round(price * random.uniform(0.98, 1.02), 2),
        "highPrice": round(price * random.uniform(1.00, 1.03), 2),
        "lowPrice":  round(price * random.uniform(0.97, 1.00), 2),
        "totalVolume": random.randint(5_000_000, 80_000_000),
        "52WkHigh": round(price * random.uniform(1.05, 1.40), 2),
        "52WkLow":  round(price * random.uniform(0.60, 0.95), 2),
    }
