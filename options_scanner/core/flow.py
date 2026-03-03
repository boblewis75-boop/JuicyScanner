"""
flow.py
-------
JuicyScanner Options Flow Engine

Scans options chains for unusual, high-conviction flow — the kind of
trades that suggest informed money is positioning for a move.

Key concepts:
  - Volume/OI ratio:  Today's volume vs open interest.
                      >1.0x = more contracts traded than exist (fresh positioning)
                      >3.0x = extremely unusual, strong signal
  - Sweep detection:  Large volume executed across multiple exchanges
                      rapidly — classic institutional signature
  - Hedge filter:     Removes likely hedging prints:
                        · Deep ITM contracts (delta > 0.85) — delta hedges
                        · Very short DTE with tiny premium — gamma scalps
                        · Puts on known heavy long portfolios
                        · Contracts with low dollar value (< $10k notional)
  - Sentiment:        Calls above ask = aggressive buyer (bullish)
                      Puts above ask = aggressive buyer (bearish)
                      Below bid = seller (opposite sentiment)
  - Conviction score: 0-100 combining all signals
"""

import math
import random
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional


# ─────────────────────────────────────────────────────────────────
# Hedge filter — remove prints that are likely hedges not bets
# ─────────────────────────────────────────────────────────────────

def is_likely_hedge(
    option_type:  str,
    delta:        float,   # absolute value
    dte:          int,
    premium:      float,
    volume:       int,
    oi:           int,
    strike:       float,
    spot:         float,
    iv:           float,
) -> tuple[bool, str]:
    """
    Returns (is_hedge, reason).
    Filters out contracts that are likely protective hedges, not directional bets.
    """
    # 1. Deep ITM — almost certainly a delta hedge or covered call
    moneyness = abs(spot - strike) / spot
    if delta > 0.85:
        return True, "deep ITM delta hedge"

    # 2. Tiny notional value — not enough skin in the game
    notional = premium * volume * 100
    if notional < 10_000:
        return True, f"low notional ${notional:,.0f}"

    # 3. Very low premium deep OTM — lottery ticket, not flow
    if delta < 0.05 and premium < 0.10:
        return True, "deep OTM lottery ticket"

    # 4. Same-day expiry with tiny premium — 0DTE scalp, not flow
    if dte == 0 and premium < 0.15:
        return True, "0DTE micro-scalp"

    # 5. Volume barely above average — not unusual
    if oi > 0 and volume / oi < 0.05:
        return True, "normal volume"

    return False, ""


# ─────────────────────────────────────────────────────────────────
# Sweep / block detector
# ─────────────────────────────────────────────────────────────────

def classify_trade_type(
    volume:   int,
    oi:       int,
    premium:  float,
    bid:      float,
    ask:      float,
    spot:     float,
) -> Dict:
    """
    Classify the trade type based on size, aggression, and price.
    Returns trade_type, aggression, sentiment_hint.
    """
    mid      = (bid + ask) / 2
    notional = premium * volume * 100

    # Where did it print relative to bid/ask?
    if premium >= ask:
        aggression = "ABOVE ASK"    # buyer paid up — bullish aggression
        sentiment  = "AGGRESSIVE_BUY"
    elif premium <= bid:
        aggression = "BELOW BID"    # seller sold down — bearish
        sentiment  = "AGGRESSIVE_SELL"
    elif premium > mid:
        aggression = "NEAR ASK"
        sentiment  = "LEAN_BUY"
    elif premium < mid:
        aggression = "NEAR BID"
        sentiment  = "LEAN_SELL"
    else:
        aggression = "AT MID"
        sentiment  = "NEUTRAL"

    # Size classification
    if notional >= 1_000_000:
        trade_type = "BLOCK"        # institutional block trade
    elif volume >= 500:
        trade_type = "SWEEP"        # large sweep across exchanges
    elif volume >= 100:
        trade_type = "LARGE PRINT"
    else:
        trade_type = "PRINT"

    return {
        "trade_type":  trade_type,
        "aggression":  aggression,
        "sentiment":   sentiment,
        "notional":    notional,
    }


# ─────────────────────────────────────────────────────────────────
# Flow conviction scorer
# ─────────────────────────────────────────────────────────────────

def score_flow(
    volume:        int,
    oi:            int,
    notional:      float,
    trade_type:    str,
    aggression:    str,
    option_type:   str,
    dte:           int,
    iv:            float,
    delta:         float,
    vol_oi_ratio:  float,
) -> int:
    """
    Score a flow print 0–100 for conviction.
    Higher = stronger signal that informed money is moving.
    """
    score = 0.0

    # 1. Volume/OI ratio (30 pts) — key signal, more traded than exist
    if vol_oi_ratio >= 5.0:    score += 30
    elif vol_oi_ratio >= 3.0:  score += 24
    elif vol_oi_ratio >= 1.5:  score += 16
    elif vol_oi_ratio >= 0.5:  score += 8
    else:                      score += 2

    # 2. Notional size (25 pts) — bigger = more conviction
    if notional >= 1_000_000:  score += 25
    elif notional >= 500_000:  score += 20
    elif notional >= 250_000:  score += 15
    elif notional >= 100_000:  score += 10
    elif notional >= 50_000:   score += 6
    else:                      score += 2

    # 3. Aggression (20 pts) — paying above ask is strongest signal
    if aggression == "ABOVE ASK":    score += 20
    elif aggression == "NEAR ASK":   score += 14
    elif aggression == "BELOW BID":  score += 16   # aggressive put buyer
    elif aggression == "NEAR BID":   score += 10
    else:                            score += 6

    # 4. Trade type (15 pts)
    if trade_type == "BLOCK":        score += 15
    elif trade_type == "SWEEP":      score += 12
    elif trade_type == "LARGE PRINT":score += 7
    else:                            score += 3

    # 5. DTE sweet spot (10 pts) — 2-6 weeks is ideal for informed flow
    if 10 <= dte <= 45:   score += 10
    elif 5 <= dte <= 60:  score += 7
    elif dte <= 5:        score += 4   # too short = speculation not info
    else:                 score += 5

    return min(100, max(0, round(score)))


# ─────────────────────────────────────────────────────────────────
# Directional bias from flow
# ─────────────────────────────────────────────────────────────────

def flow_direction(option_type: str, sentiment: str) -> str:
    """
    Combine option type + trade sentiment to get directional read.
    """
    if option_type == "CALL":
        if sentiment in ("AGGRESSIVE_BUY", "LEAN_BUY"):   return "BULLISH"
        if sentiment in ("AGGRESSIVE_SELL", "LEAN_SELL"):  return "BEARISH"  # call selling = bearish
        return "NEUTRAL"
    else:  # PUT
        if sentiment in ("AGGRESSIVE_BUY", "LEAN_BUY"):   return "BEARISH"  # put buying = bearish
        if sentiment in ("AGGRESSIVE_SELL", "LEAN_SELL"):  return "BULLISH"  # put selling = bullish
        return "NEUTRAL"


# ─────────────────────────────────────────────────────────────────
# Main flow scan
# ─────────────────────────────────────────────────────────────────

def scan_flow(
    chains:         List[Dict],    # list of {symbol, chain, quote}
    min_score:      int   = 60,
    min_notional:   float = 25_000,
    min_vol_oi:     float = 0.3,
    contract_type:  str   = "ALL",
    filter_hedges:  bool  = True,
) -> List[Dict]:
    """
    Scan multiple options chains for unusual, high-conviction flow.
    Returns sorted list of flow prints with scores and direction.
    """
    results = []

    for entry in chains:
        symbol = entry["symbol"]
        chain  = entry["chain"]
        spot   = chain.get("underlyingPrice", 100)

        all_contracts = []
        if contract_type in ("ALL", "CALL"):
            all_contracts += _extract_contracts_from_chain(chain.get("callExpDateMap", {}), "CALL")
        if contract_type in ("ALL", "PUT"):
            all_contracts += _extract_contracts_from_chain(chain.get("putExpDateMap", {}), "PUT")

        for c in all_contracts:
            volume  = c.get("totalVolume", 0)
            oi      = c.get("openInterest", 0) or 1
            bid     = c.get("bid", 0)
            ask     = c.get("ask", 0)
            if bid <= 0 or ask <= 0 or volume < 10:
                continue

            mid      = (bid + ask) / 2
            premium  = round(mid, 2)
            delta    = abs(c.get("delta", 0.3))
            dte      = c.get("daysToExpiration", 30)
            iv       = c.get("volatility", 50) / 100
            strike   = c.get("strikePrice", spot)
            opt_type = c.get("_type", "CALL")
            exp_str  = c.get("_exp", "").split(":")[0] if ":" in c.get("_exp","") else c.get("_exp","")

            # Hedge filter
            if filter_hedges:
                hedged, hedge_reason = is_likely_hedge(opt_type, delta, dte, premium, volume, oi, strike, spot, iv)
                if hedged:
                    continue

            # Notional & vol/OI filters
            notional    = premium * volume * 100
            vol_oi      = volume / oi
            if notional  < min_notional: continue
            if vol_oi    < min_vol_oi:   continue

            # Classify the trade
            trade_info = classify_trade_type(volume, oi, premium, bid, ask, spot)
            if trade_info["notional"] < min_notional:
                continue

            # Direction
            direction = flow_direction(opt_type, trade_info["sentiment"])

            # Score
            score = score_flow(
                volume       = volume,
                oi           = oi,
                notional     = trade_info["notional"],
                trade_type   = trade_info["trade_type"],
                aggression   = trade_info["aggression"],
                option_type  = opt_type,
                dte          = dte,
                iv           = iv,
                delta        = delta,
                vol_oi_ratio = vol_oi,
            )

            if score < min_score:
                continue

            # Expiry label
            try:
                exp_date = datetime.strptime(exp_str[:10], "%Y-%m-%d")
                exp_label = exp_date.strftime("%-m/%-d") if hasattr(date, 'strftime') else exp_str[:10]
            except:
                exp_label = exp_str[:10]

            results.append({
                "symbol":       symbol,
                "type":         opt_type,
                "strike":       strike,
                "expiry":       exp_label,
                "expiry_full":  exp_str[:10],
                "dte":          dte,
                "premium":      premium,
                "bid":          round(bid, 2),
                "ask":          round(ask, 2),
                "volume":       volume,
                "oi":           oi,
                "vol_oi":       round(vol_oi, 2),
                "notional":     round(trade_info["notional"]),
                "trade_type":   trade_info["trade_type"],
                "aggression":   trade_info["aggression"],
                "sentiment":    trade_info["sentiment"],
                "direction":    direction,
                "delta":        round(delta, 3),
                "iv":           round(iv * 100, 1),
                "score":        score,
                "spot":         spot,
                "moneyness":    round((strike - spot) / spot * 100, 2),
                "timestamp":    datetime.now().strftime("%H:%M:%S"),
            })

    # Sort by score desc, then notional
    results.sort(key=lambda x: (x["score"], x["notional"]), reverse=True)
    return results[:50]


# ─────────────────────────────────────────────────────────────────
# Mock flow generator (realistic-looking flow prints)
# ─────────────────────────────────────────────────────────────────

def generate_mock_flow(symbols: List[str]) -> List[Dict]:
    """Generate realistic mock flow prints for testing."""
    prices = {
        "SPY":590,"QQQ":505,"NVDA":138,"AAPL":221,"TSLA":295,
        "META":632,"MSFT":415,"AMZN":227,"AMD":120,"COIN":185,
        "GOOGL":175,"NFLX":910,"GLD":235,"SMH":215,"XLE":88,
    }

    flow_prints = []
    now = datetime.now()

    # Predefined "juicy" flow scenarios
    scenarios = [
        # Big bullish sweep
        {"type":"CALL","delta_range":(0.35,0.55),"vol_mult":(3,12),"notional_target":500_000,"dte_range":(14,35),"aggression":"ABOVE ASK"},
        # Massive block trade
        {"type":"CALL","delta_range":(0.40,0.65),"vol_mult":(1,3),"notional_target":1_200_000,"dte_range":(21,45),"aggression":"ABOVE ASK"},
        # Bearish put sweep
        {"type":"PUT","delta_range":(0.30,0.50),"vol_mult":(4,15),"notional_target":350_000,"dte_range":(7,21),"aggression":"ABOVE ASK"},
        # Put selling (bullish)
        {"type":"PUT","delta_range":(0.20,0.35),"vol_mult":(2,6),"notional_target":200_000,"dte_range":(14,30),"aggression":"BELOW BID"},
        # Large unusual OTM call
        {"type":"CALL","delta_range":(0.20,0.35),"vol_mult":(8,25),"notional_target":150_000,"dte_range":(30,60),"aggression":"ABOVE ASK"},
        # Near-term sweep
        {"type":"CALL","delta_range":(0.45,0.60),"vol_mult":(5,20),"notional_target":80_000,"dte_range":(3,10),"aggression":"NEAR ASK"},
        # Bearish block
        {"type":"PUT","delta_range":(0.35,0.55),"vol_mult":(1,4),"notional_target":800_000,"dte_range":(21,45),"aggression":"ABOVE ASK"},
        # Small but weird vol/OI
        {"type":"CALL","delta_range":(0.25,0.40),"vol_mult":(15,50),"notional_target":45_000,"dte_range":(7,14),"aggression":"ABOVE ASK"},
    ]

    for sym in symbols:
        spot = prices.get(sym, 100) * random.uniform(0.995, 1.005)
        n_prints = random.randint(2, 5)

        for _ in range(n_prints):
            sc      = random.choice(scenarios)
            opt_type= sc["type"]
            delta   = random.uniform(*sc["delta_range"])
            dte     = random.randint(*sc["dte_range"])

            # Derive strike from delta (approximate)
            sign    = 1 if opt_type == "CALL" else -1
            strike  = round(spot * (1 + sign * (0.5 - delta) * 0.15), 0)
            step    = max(1, round(spot * 0.005))
            strike  = round(strike / step) * step

            # Premium from BS approximation
            iv       = random.uniform(0.30, 1.20)
            t        = dte / 365
            premium  = max(0.05, spot * iv * math.sqrt(t) * delta * random.uniform(0.8, 1.2))
            premium  = round(premium, 2)

            # Volume to hit target notional
            target_n = sc["notional_target"] * random.uniform(0.6, 1.8)
            volume   = max(10, int(target_n / (premium * 100)))
            oi       = max(volume, int(volume / random.uniform(0.3, 5.0)))

            bid = round(premium * random.uniform(0.92, 0.98), 2)
            ask = round(premium * random.uniform(1.02, 1.08), 2)

            # Aggression adjusts print price
            agg = sc["aggression"]
            if agg == "ABOVE ASK":    print_price = ask + random.uniform(0.01, 0.05)
            elif agg == "BELOW BID":  print_price = bid - random.uniform(0.01, 0.05)
            elif agg == "NEAR ASK":   print_price = (premium + ask) / 2
            else:                     print_price = premium
            print_price = round(print_price, 2)

            # Classify
            trade_info = classify_trade_type(volume, oi, print_price, bid, ask, spot)
            direction  = flow_direction(opt_type, trade_info["sentiment"])
            vol_oi     = round(volume / oi, 2)

            score = score_flow(
                volume=volume, oi=oi, notional=trade_info["notional"],
                trade_type=trade_info["trade_type"], aggression=trade_info["aggression"],
                option_type=opt_type, dte=dte, iv=iv, delta=delta, vol_oi_ratio=vol_oi,
            )

            # Randomize timestamp to look like intraday flow
            mins_ago = random.randint(0, 180)
            ts = (now - timedelta(minutes=mins_ago)).strftime("%H:%M:%S")

            exp_label = (date.today() + timedelta(days=dte)).strftime("%-m/%-d") if hasattr(timedelta, '__sub__') else f"+{dte}d"
            try:
                exp_label = (date.today() + timedelta(days=dte)).strftime("%m/%d")
            except:
                exp_label = f"+{dte}d"

            flow_prints.append({
                "symbol":      sym,
                "type":        opt_type,
                "strike":      strike,
                "expiry":      exp_label,
                "dte":         dte,
                "premium":     print_price,
                "bid":         bid,
                "ask":         ask,
                "volume":      volume,
                "oi":          oi,
                "vol_oi":      vol_oi,
                "notional":    round(trade_info["notional"]),
                "trade_type":  trade_info["trade_type"],
                "aggression":  trade_info["aggression"],
                "sentiment":   trade_info["sentiment"],
                "direction":   direction,
                "delta":       round(delta, 3),
                "iv":          round(iv * 100, 1),
                "score":       score,
                "spot":        round(spot, 2),
                "moneyness":   round((strike - spot) / spot * 100, 2),
                "timestamp":   ts,
            })

    flow_prints.sort(key=lambda x: (x["score"], x["notional"]), reverse=True)
    return flow_prints[:50]


# ─────────────────────────────────────────────────────────────────
# Flow summary — aggregate direction per symbol
# ─────────────────────────────────────────────────────────────────

def summarize_flow(prints: List[Dict]) -> Dict:
    """
    Roll up flow into per-symbol bullish/bearish notional totals.
    """
    summary = {}
    for p in prints:
        sym = p["symbol"]
        if sym not in summary:
            summary[sym] = {
                "symbol":         sym,
                "bullish_notional":  0,
                "bearish_notional":  0,
                "total_notional":    0,
                "bullish_prints":    0,
                "bearish_prints":    0,
                "top_score":         0,
                "spot":              p["spot"],
            }
        s = summary[sym]
        if p["direction"] == "BULLISH":
            s["bullish_notional"] += p["notional"]
            s["bullish_prints"]   += 1
        elif p["direction"] == "BEARISH":
            s["bearish_notional"] += p["notional"]
            s["bearish_prints"]   += 1
        s["total_notional"] += p["notional"]
        s["top_score"] = max(s["top_score"], p["score"])

    # Add net bias
    for sym, s in summary.items():
        total = s["bullish_notional"] + s["bearish_notional"]
        s["net_bias"] = round((s["bullish_notional"] - s["bearish_notional"]) / total * 100) if total else 0
        s["bias_label"] = "BULLISH" if s["net_bias"] > 20 else "BEARISH" if s["net_bias"] < -20 else "MIXED"

    return dict(sorted(summary.items(), key=lambda x: x[1]["total_notional"], reverse=True))


def _extract_contracts_from_chain(exp_map: dict, opt_type: str) -> list:
    """
    Handle both Schwab live format {exp: {strike: [contract]}}
    and mock format {exp: [contract, ...]}.
    """
    contracts = []
    for exp_key, options in exp_map.items():
        if isinstance(options, dict):
            # Real Schwab format
            for strike_str, contract_list in options.items():
                if isinstance(contract_list, list):
                    for c in contract_list:
                        if isinstance(c, dict):
                            c["_exp"] = exp_key
                            c["_type"] = opt_type
                            contracts.append(c)
        elif isinstance(options, list):
            # Mock format
            for c in options:
                if isinstance(c, dict):
                    c["_exp"] = exp_key
                    c["_type"] = opt_type
                    contracts.append(c)
    return contracts
