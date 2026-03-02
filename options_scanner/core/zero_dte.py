"""
zero_dte.py
-----------
JuicyScanner 0DTE Engine

0DTE options are fundamentally different from regular options:
  - Gamma is massive near ATM — small moves = huge % swings
  - Theta decays to zero by end of day — time is the enemy
  - IV spikes on entry, collapses fast
  - Target is a quick intraday move (0.3–2%) not a multi-day trend
  - Risk management is entirely different: tighter stops, faster targets
  - Best setups: opening drive, VWAP reclaims, key level bounces

Scoring model is purpose-built for 0DTE:
  - High delta (0.45-0.60) preferred — need quick ITM conversion
  - Low spread % — critical because premium is already tiny
  - High IV rank acceptable — premium is cheap relative to gamma
  - Tech momentum indicators weighted MORE than usual
  - Probability model adjusted for intraday time windows (minutes, not days)
"""

import math
from typing import List, Dict, Any, Optional
from datetime import datetime, time as dt_time


# ─────────────────────────────────────────────────────────────────
# Market session helpers
# ─────────────────────────────────────────────────────────────────

# Best windows for 0DTE entries (Eastern time)
WINDOWS = [
    {"name": "Opening Drive",    "start": dt_time(9, 30),  "end": dt_time(10, 15), "quality": "PRIME"},
    {"name": "Mid-Morning",      "start": dt_time(10, 15), "end": dt_time(11, 30), "quality": "GOOD"},
    {"name": "Lunch Chop",       "start": dt_time(11, 30), "end": dt_time(13, 0),  "quality": "AVOID"},
    {"name": "Afternoon Trend",  "start": dt_time(13, 0),  "end": dt_time(15, 0),  "quality": "GOOD"},
    {"name": "Power Hour",       "start": dt_time(15, 0),  "end": dt_time(15, 45), "quality": "PRIME"},
    {"name": "Last 15 min",      "start": dt_time(15, 45), "end": dt_time(16, 0),  "quality": "AVOID"},
]


def get_session_window(now: Optional[dt_time] = None) -> Dict:
    """Return current market window and quality rating."""
    if now is None:
        now = datetime.now().time()
    for w in WINDOWS:
        if w["start"] <= now < w["end"]:
            return w
    if now < dt_time(9, 30):
        return {"name": "Pre-Market",  "start": dt_time(0,0), "end": dt_time(9,30),  "quality": "CLOSED"}
    return {"name": "After-Hours",  "start": dt_time(16,0), "end": dt_time(23,59), "quality": "CLOSED"}


def minutes_until_close(now: Optional[dt_time] = None) -> int:
    """Minutes remaining in the trading day."""
    if now is None:
        now = datetime.now().time()
    close = dt_time(16, 0)
    if now >= close:
        return 0
    now_mins   = now.hour * 60 + now.minute
    close_mins = 16 * 60
    return max(0, close_mins - now_mins)


# ─────────────────────────────────────────────────────────────────
# 0DTE intraday probability model
# ─────────────────────────────────────────────────────────────────

def calc_0dte_probability(
    option_type: str,       # CALL / PUT
    underlying_price: float,
    strike: float,
    premium: float,
    iv: float,              # annual IV as decimal
    delta: float,           # positive
    target_pct: float,      # % move needed (positive)
    tech_score: float,      # 0-100 bull score
    minutes_left: int,      # minutes until market close
) -> Dict:
    """
    0DTE-specific probability model.
    
    Key differences from regular options:
      - Time horizon is MINUTES not days
      - We care about touching the target today, not at expiry
      - Gamma scalping window matters
      - Delta is the dominant prob signal at 0DTE
    """
    t_hours = max(minutes_left / 60, 0.083)  # minimum 5min window
    t_years = t_hours / 8760

    # ── Intraday IV conversion ──────────────────────────────────
    # Annual IV → intraday expected move
    intraday_sigma = iv * math.sqrt(t_years)
    expected_move_pct = intraday_sigma * 100

    # ── Model 1: Delta (strongest signal at 0DTE) ───────────────
    delta_prob = abs(delta) * 100

    # ── Model 2: Intraday move probability ──────────────────────
    # P(move >= target%) given intraday IV window
    if expected_move_pct > 0.001:
        z = target_pct / expected_move_pct
        hist_prob = (1 - _norm_cdf(z)) * 100
    else:
        hist_prob = 10.0

    # ── Model 3: VWAP/momentum technical score ──────────────────
    tech_prob = tech_score  # already 0-100

    # ── Model 4: Moneyness — how close to ATM? ──────────────────
    # ATM strikes have highest gamma and highest 0DTE win rate
    moneyness = abs(underlying_price - strike) / underlying_price * 100
    if moneyness < 0.5:
        atm_bonus = 20
    elif moneyness < 1.0:
        atm_bonus = 12
    elif moneyness < 2.0:
        atm_bonus = 5
    else:
        atm_bonus = 0

    # ── Time window penalty ─────────────────────────────────────
    if minutes_left < 30:
        time_penalty = -20   # too late, theta nuke
    elif minutes_left < 60:
        time_penalty = -8
    elif minutes_left > 300:
        time_penalty = -5    # too early, direction unclear
    else:
        time_penalty = 0

    # ── Weighted blend ───────────────────────────────────────────
    raw = (
        delta_prob * 0.40 +
        hist_prob  * 0.25 +
        tech_prob  * 0.25 +
        atm_bonus  * 0.10
    ) + time_penalty

    likelihood = round(min(94, max(5, raw)), 1)
    confidence = "HIGH" if likelihood >= 68 else "MEDIUM" if likelihood >= 50 else "LOW"

    return {
        "likelihood_pct":    likelihood,
        "delta_prob":        round(delta_prob, 1),
        "intraday_prob":     round(hist_prob, 1),
        "tech_confirmation": round(tech_prob, 1),
        "atm_bonus":         atm_bonus,
        "time_penalty":      time_penalty,
        "expected_move_pct": round(expected_move_pct, 2),
        "confidence":        confidence,
        "minutes_left":      minutes_left,
    }


# ─────────────────────────────────────────────────────────────────
# 0DTE scoring engine
# ─────────────────────────────────────────────────────────────────

def score_0dte_play(
    premium: float,
    rr: float,
    delta: float,
    spread_pct: float,
    iv: float,
    oi: int,
    volume: int,
    target_pct: float,
    likelihood_pct: float,
    tech_bias: float,
    option_type: str,
    window_quality: str,
    minutes_left: int,
    moneyness_pct: float,
) -> int:
    """
    0DTE-specific scoring. Weights are tuned for intraday plays.
    Total: 100 pts
    """
    score = 0.0

    # 1. LIKELIHOOD (35 pts) — king for 0DTE, must be high probability
    if likelihood_pct >= 75:   score += 35
    elif likelihood_pct >= 65: score += 28
    elif likelihood_pct >= 55: score += 20
    elif likelihood_pct >= 45: score += 12
    else:                      score += 4

    # 2. DELTA (20 pts) — 0DTE needs 0.40-0.60 for quick gamma pop
    if 0.45 <= delta <= 0.60:   score += 20
    elif 0.38 <= delta <= 0.65: score += 14
    elif 0.30 <= delta <= 0.72: score += 8
    else:                       score += 2

    # 3. TECH BIAS (20 pts) — momentum is everything intraday
    bias_pts = abs(tech_bias)
    if bias_pts >= 60:   score += 20
    elif bias_pts >= 40: score += 14
    elif bias_pts >= 20: score += 8
    else:                score += 2

    # 4. SPREAD QUALITY (10 pts) — tight spread = fast in/out
    if spread_pct < 0.04:   score += 10
    elif spread_pct < 0.08: score += 7
    elif spread_pct < 0.12: score += 4
    else:                   score += 1

    # 5. R:R RATIO (8 pts)
    if rr >= 3.0:   score += 8
    elif rr >= 2.0: score += 5
    else:           score += 2

    # 6. WINDOW QUALITY (7 pts)
    if window_quality == "PRIME":  score += 7
    elif window_quality == "GOOD": score += 4
    elif window_quality == "AVOID": score += 0

    # Bonus: high volume (liquidity confirmation)
    if volume > 5000:  score += 2
    if volume > 15000: score += 1  # stacking bonus for very liquid

    # Penalty: not enough time left
    if minutes_left < 30: score -= 25
    elif minutes_left < 60: score -= 10

    return min(100, max(0, round(score)))


# ─────────────────────────────────────────────────────────────────
# 0DTE R:R calculation (different from regular options)
# ─────────────────────────────────────────────────────────────────

def calc_0dte_rr(
    premium: float,
    target_pct: float,    # % move in underlying
    delta: float,
    underlying_price: float,
) -> Dict:
    """
    0DTE R:R is about the option's % gain on a given underlying move.
    
    If underlying moves target_pct:
      - Option gains ≈ delta × underlying_move × 100 (per contract)
      - Gain % on premium = (delta × underlying_move) / premium × 100
    
    Stop: typically 50-60% of premium (faster than regular options)
    """
    underlying_move = underlying_price * target_pct / 100
    expected_gain   = delta * underlying_move  # per-share gain
    gain_pct        = (expected_gain / premium) * 100 if premium > 0 else 0
    stop_pct        = -50.0   # stop at 50% of premium
    rr              = abs(gain_pct / abs(stop_pct)) if stop_pct != 0 else 0

    return {
        "rr":           round(rr, 2),
        "gain_pct":     round(gain_pct, 1),
        "stop_pct":     stop_pct,
        "target_gain":  round(expected_gain, 2),
    }


# ─────────────────────────────────────────────────────────────────
# Main 0DTE scan function
# ─────────────────────────────────────────────────────────────────

def scan_0dte(
    chain: Dict,
    quote: Dict,
    tech: Dict,
    min_score: int    = 65,
    min_likelihood: float = 55.0,
    contract_type: str = "ALL",
    now_time: Optional[dt_time] = None,
) -> List[Dict]:
    """
    Scan an options chain for 0DTE plays only.
    Returns plays sorted by score descending.
    
    Key filter: DTE == 0 (expiring today)
    """
    symbol           = chain["symbol"]
    underlying_price = chain["underlyingPrice"]
    window           = get_session_window(now_time)
    mins_left        = minutes_until_close(now_time)
    tech_bias        = tech.get("bias_score", 0)
    tech_bull        = tech.get("bull_score", 50)

    # Don't surface plays during avoid windows
    if window["quality"] == "AVOID" and mins_left > 0:
        return []
    if window["quality"] == "CLOSED":
        return []

    results = []
    all_contracts = []

    def _extract_0dte(exp_map):
        out = []
        for exp, opts in exp_map.items():
            # Real Schwab: {strike: [contract]}
            if isinstance(opts, dict):
                for strike_str, clist in opts.items():
                    items = clist if isinstance(clist, list) else [clist]
                    for o in items:
                        if isinstance(o, dict) and o.get("daysToExpiration", 99) == 0:
                            out.append(o)
            elif isinstance(opts, list):
                for o in opts:
                    if isinstance(o, dict) and o.get("daysToExpiration", 99) == 0:
                        out.append(o)
        return out

    # Collect 0DTE contracts only (DTE == 0)
    if contract_type in ("ALL", "CALL"):
        all_contracts += _extract_0dte(chain.get("callExpDateMap", {}))
    if contract_type in ("ALL", "PUT"):
        all_contracts += _extract_0dte(chain.get("putExpDateMap", {}))

    for c in all_contracts:
        bid = c.get("bid", 0)
        ask = c.get("ask", 0)
        if bid <= 0 or ask <= 0:
            continue

        mid         = (bid + ask) / 2
        spread_pct  = (ask - bid) / mid if mid > 0 else 999
        premium     = round(mid, 2)
        delta       = abs(c.get("delta", 0))
        oi          = c.get("openInterest", 0)
        volume      = c.get("totalVolume", 0)
        iv          = c.get("volatility", 50) / 100
        strike      = c.get("strikePrice", 0)
        opt_type    = c.get("putCall", "CALL")
        moneyness   = abs(underlying_price - strike) / underlying_price * 100

        # Hard filters for 0DTE
        if spread_pct > 0.20:      continue   # max 20% spread
        if oi < 100:               continue   # some liquidity required
        if delta < 0.20:           continue   # too far OTM
        if delta > 0.85:           continue   # too deep ITM
        if moneyness > 3.0:        continue   # must be within 3% of ATM

        # Target: 50% gain on premium (conservative for 0DTE)
        target_gain_pct   = 50.0              # target 50% premium gain
        underlying_move_needed = (target_gain_pct / 100 * premium) / delta
        target_pct_needed = underlying_move_needed / underlying_price * 100

        # R:R
        rr_data = calc_0dte_rr(premium, target_pct_needed, delta, underlying_price)
        rr      = rr_data["rr"]

        # Probability
        tech_score = tech_bull if opt_type == "CALL" else (100 - tech_bull)
        prob = calc_0dte_probability(
            option_type      = opt_type,
            underlying_price = underlying_price,
            strike           = strike,
            premium          = premium,
            iv               = iv,
            delta            = delta,
            target_pct       = target_pct_needed,
            tech_score       = tech_score,
            minutes_left     = mins_left,
        )

        if prob["likelihood_pct"] < min_likelihood:
            continue

        # Score
        score = score_0dte_play(
            premium        = premium,
            rr             = rr,
            delta          = delta,
            spread_pct     = spread_pct,
            iv             = iv,
            oi             = oi,
            volume         = volume,
            target_pct     = target_pct_needed,
            likelihood_pct = prob["likelihood_pct"],
            tech_bias      = tech_bias if opt_type == "CALL" else -tech_bias,
            option_type    = opt_type,
            window_quality = window["quality"],
            minutes_left   = mins_left,
            moneyness_pct  = moneyness,
        )

        if score < min_score:
            continue

        results.append({
            "ticker":          symbol,
            "type":            opt_type,
            "strike":          strike,
            "expiry":          "0DTE",
            "dte":             0,
            "premium":         premium,
            "bid":             bid,
            "ask":             ask,
            "spread_pct":      round(spread_pct * 100, 2),
            "rr":              rr,
            "target_pct":      round(target_pct_needed, 2),
            "stop_pct":        -50.0,
            "delta":           round(c.get("delta", 0), 3),
            "gamma":           round(c.get("gamma", 0), 4),
            "theta":           round(c.get("theta", 0), 4),
            "iv":              round(c.get("volatility", 0), 1),
            "oi":              oi,
            "volume":          volume,
            "moneyness_pct":   round(moneyness, 2),
            "underlying_px":   underlying_price,
            "likelihood":      prob,
            "score":           score,
            "window":          window,
            "minutes_left":    mins_left,
            "tech_bias":       tech_bias,
            "alert_type":      _classify_alert(score, prob["likelihood_pct"], window["quality"]),
            "rationale":       _rationale_0dte(
                symbol, opt_type, strike, underlying_price,
                score, prob, rr, window, mins_left, tech_bias
            ),
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:10]  # top 10 only — 0DTE needs focus


# ─────────────────────────────────────────────────────────────────
# Alert classification
# ─────────────────────────────────────────────────────────────────

def _classify_alert(score: int, likelihood: float, window: str) -> str:
    """Classify the urgency of a 0DTE alert."""
    if score >= 80 and likelihood >= 68 and window == "PRIME":
        return "🔥 PRIME SETUP"
    elif score >= 72 and likelihood >= 60:
        return "⚡ HIGH QUALITY"
    elif score >= 65:
        return "✅ VALID PLAY"
    else:
        return "📊 MONITOR"


# ─────────────────────────────────────────────────────────────────
# Rationale
# ─────────────────────────────────────────────────────────────────

def _rationale_0dte(
    symbol, opt_type, strike, underlying_price,
    score, prob, rr, window, mins_left, tech_bias
) -> str:
    direction = "bullish" if opt_type == "CALL" else "bearish"
    conf      = prob["confidence"]
    lh        = prob["likelihood_pct"]
    atm_note  = "Near-ATM strike gives maximum gamma exposure." if abs(underlying_price - strike) / underlying_price < 0.01 else f"Strike ${strike} is {abs(underlying_price-strike)/underlying_price*100:.1f}% from current price."
    window_note = f"Entering during {window['name']} ({window['quality']} quality window, {mins_left}min left)."
    tech_note = "Strong directional momentum supports entry." if abs(tech_bias) >= 40 else "Technical signals are moderate — watch for confirmation."
    return f"{symbol} 0DTE {direction} setup. {atm_note} {prob['likelihood_pct']:.0f}% likelihood ({conf} confidence). R:R {rr}:1. {window_note} {tech_note}"


# ─────────────────────────────────────────────────────────────────
# Mock 0DTE chain generator (for testing without live API)
# ─────────────────────────────────────────────────────────────────

def generate_mock_0dte_chain(symbol: str) -> Dict:
    """Generate a realistic 0DTE options chain (DTE=0 contracts only)."""
    import random
    prices = {
        "SPY": 590.0, "QQQ": 505.0, "SPX": 5900.0,
        "NVDA": 138.0, "AAPL": 221.5, "TSLA": 295.0,
        "AMZN": 227.0, "META": 632.0, "MSFT": 415.0,
    }
    price = prices.get(symbol, 100.0) * random.uniform(0.995, 1.005)
    today = datetime.now().strftime("%Y-%m-%d")

    calls, puts = [], []
    for offset in [-0.025, -0.015, -0.005, 0, 0.005, 0.015, 0.025]:
        strike  = round(price * (1 + offset), 0)
        # 0DTE has very high gamma, low time value
        intrinsic_call = max(0, price - strike)
        intrinsic_put  = max(0, strike - price)
        time_val   = abs(price - strike) * 0.02 + random.uniform(0.10, 0.80)
        call_mid   = round(max(0.05, intrinsic_call + time_val), 2)
        put_mid    = round(max(0.05, intrinsic_put + time_val), 2)
        spread     = round(call_mid * random.uniform(0.03, 0.09), 2)
        delta_c    = max(0.05, min(0.95, 0.50 - offset * 8 + random.uniform(-0.05, 0.05)))
        iv_today   = random.uniform(0.40, 1.20)  # 0DTE IV spikes

        calls.append({
            "strikePrice": strike, "expirationDate": today, "daysToExpiration": 0,
            "bid": round(call_mid - spread/2, 2), "ask": round(call_mid + spread/2, 2), "mark": call_mid,
            "delta": round(delta_c, 3), "gamma": round(random.uniform(0.05, 0.25), 4),
            "theta": round(-random.uniform(0.30, 2.0), 4), "vega": round(random.uniform(0.01, 0.08), 4),
            "volatility": round(iv_today * 100, 2),
            "openInterest": random.randint(500, 50000),
            "totalVolume":  random.randint(200, 30000),
            "inTheMoney": price > strike, "putCall": "CALL",
        })
        puts.append({
            "strikePrice": strike, "expirationDate": today, "daysToExpiration": 0,
            "bid": round(put_mid - spread/2, 2), "ask": round(put_mid + spread/2, 2), "mark": put_mid,
            "delta": round(delta_c - 1, 3), "gamma": round(random.uniform(0.05, 0.25), 4),
            "theta": round(-random.uniform(0.30, 2.0), 4), "vega": round(random.uniform(0.01, 0.08), 4),
            "volatility": round(iv_today * 100, 2),
            "openInterest": random.randint(500, 40000),
            "totalVolume":  random.randint(200, 25000),
            "inTheMoney": price < strike, "putCall": "PUT",
        })

    return {
        "symbol": symbol,
        "underlyingPrice": round(price, 2),
        "callExpDateMap": {today: calls},
        "putExpDateMap":  {today: puts},
        "status": "SUCCESS",
    }


def _norm_cdf(x: float) -> float:
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
