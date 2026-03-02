"""
technicals.py
-------------
Technical Analysis Engine for OptionsEdge AI.

Computes the following indicators on a price series:
  - RSI (14-period)
  - MACD (12/26/9 EMA)
  - Moving Averages (20, 50, 200 EMA/SMA)
  - Volume Surge detection
  - Price Channel detection (support/resistance, trend channels)
  - Bollinger Bands
  - Overall Directional Bias score (-100 to +100)
  - Bull/Bear signal summary

In mock mode, generates realistic synthetic OHLCV data per symbol.
In live mode, plug in real OHLCV bars from Schwab's price history endpoint.
"""

import math
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────
# Data structure
# ─────────────────────────────────────────────────────────────────

@dataclass
class Bar:
    """Single OHLCV bar."""
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


# ─────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────

def analyze(bars: List[Bar], symbol: str = "") -> Dict[str, Any]:
    """
    Run full technical analysis on a list of OHLCV bars.
    Returns a dict with all indicators + a directional bias score.
    Requires at least 50 bars for full analysis (200 recommended).
    """
    if len(bars) < 14:
        return {"error": "Not enough bars for analysis", "bias_score": 0}

    closes  = [b.close for b in bars]
    highs   = [b.high  for b in bars]
    lows    = [b.low   for b in bars]
    volumes = [b.volume for b in bars]

    # ── Indicators ───────────────────────────────────────────────
    rsi_val   = calc_rsi(closes, period=14)
    macd_data = calc_macd(closes)
    ma_data   = calc_moving_averages(closes)
    bb_data   = calc_bollinger(closes)
    vol_data  = calc_volume_surge(volumes)
    channel   = detect_channel(highs, lows, closes)

    # ── Directional bias score (-100 to +100) ────────────────────
    bias, signals = calc_bias_score(
        rsi=rsi_val,
        macd=macd_data,
        mas=ma_data,
        volume=vol_data,
        channel=channel,
        bb=bb_data,
        current_price=closes[-1],
    )

    # ── Overall tech score for probability engine (0-100) ────────
    # Convert bias to 0-100 scale, centered at 50 for neutral
    bull_score = round(50 + bias / 2, 1)

    return {
        "symbol":          symbol,
        "current_price":   round(closes[-1], 2),
        "bias_score":      bias,           # -100 (strong bear) to +100 (strong bull)
        "bull_score":      bull_score,     # 0-100, used by probability engine
        "signals":         signals,        # list of signal dicts
        "rsi":             rsi_val,
        "macd":            macd_data,
        "moving_averages": ma_data,
        "bollinger":       bb_data,
        "volume":          vol_data,
        "channel":         channel,
        "bar_count":       len(bars),
    }


# ─────────────────────────────────────────────────────────────────
# RSI
# ─────────────────────────────────────────────────────────────────

def calc_rsi(closes: List[float], period: int = 14) -> Dict[str, Any]:
    if len(closes) < period + 1:
        return {"value": 50, "signal": "NEUTRAL", "note": "Insufficient data"}

    gains, losses = [], []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i-1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))

    # Wilder smoothing
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    rs  = avg_gain / avg_loss if avg_loss > 0 else 100
    rsi = round(100 - (100 / (1 + rs)), 2)

    # Signal
    if rsi >= 70:
        signal = "OVERBOUGHT"
        note   = f"RSI {rsi:.1f} — overbought, potential pullback or reversal ahead."
    elif rsi <= 30:
        signal = "OVERSOLD"
        note   = f"RSI {rsi:.1f} — oversold, potential bounce or reversal ahead."
    elif 50 < rsi < 70:
        signal = "BULLISH"
        note   = f"RSI {rsi:.1f} — in bullish momentum zone."
    elif 30 < rsi <= 50:
        signal = "BEARISH"
        note   = f"RSI {rsi:.1f} — in bearish momentum zone."
    else:
        signal = "NEUTRAL"
        note   = f"RSI {rsi:.1f} — neutral."

    return {"value": rsi, "signal": signal, "note": note}


# ─────────────────────────────────────────────────────────────────
# MACD
# ─────────────────────────────────────────────────────────────────

def calc_macd(
    closes: List[float],
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> Dict[str, Any]:
    if len(closes) < slow + signal_period:
        return {"macd_line": 0, "signal_line": 0, "histogram": 0, "signal": "NEUTRAL", "note": "Insufficient data"}

    ema_fast   = _ema(closes, fast)
    ema_slow   = _ema(closes, slow)
    macd_line  = [f - s for f, s in zip(ema_fast[slow-fast:], ema_slow)]
    signal_line = _ema(macd_line, signal_period)
    histogram  = [m - s for m, s in zip(macd_line[signal_period-1:], signal_line)]

    # Latest values
    cur_macd  = round(macd_line[-1], 4)
    cur_sig   = round(signal_line[-1], 4)
    cur_hist  = round(histogram[-1], 4)
    prev_hist = round(histogram[-2], 4) if len(histogram) >= 2 else 0

    # Crossover detection
    prev_macd = macd_line[-2] if len(macd_line) >= 2 else cur_macd
    prev_sig  = signal_line[-2] if len(signal_line) >= 2 else cur_sig

    crossed_up   = prev_macd <= prev_sig and cur_macd > cur_sig
    crossed_down = prev_macd >= prev_sig and cur_macd < cur_sig
    hist_growing = cur_hist > prev_hist

    if crossed_up:
        signal = "BULLISH_CROSS"
        note   = "MACD crossed above signal line — bullish momentum shift."
    elif crossed_down:
        signal = "BEARISH_CROSS"
        note   = "MACD crossed below signal line — bearish momentum shift."
    elif cur_macd > cur_sig and hist_growing:
        signal = "BULLISH"
        note   = f"MACD above signal ({cur_macd:.3f} > {cur_sig:.3f}), histogram expanding — bullish."
    elif cur_macd < cur_sig and not hist_growing:
        signal = "BEARISH"
        note   = f"MACD below signal ({cur_macd:.3f} < {cur_sig:.3f}), histogram contracting — bearish."
    else:
        signal = "NEUTRAL"
        note   = f"MACD neutral (line: {cur_macd:.3f}, signal: {cur_sig:.3f})."

    return {
        "macd_line":   cur_macd,
        "signal_line": cur_sig,
        "histogram":   cur_hist,
        "signal":      signal,
        "note":        note,
        "bullish_cross": crossed_up,
        "bearish_cross": crossed_down,
        "hist_series": [round(h, 4) for h in histogram[-20:]],  # last 20 bars
    }


# ─────────────────────────────────────────────────────────────────
# Moving Averages
# ─────────────────────────────────────────────────────────────────

def calc_moving_averages(closes: List[float]) -> Dict[str, Any]:
    price = closes[-1]
    result = {}

    periods = {"ema20": 20, "ema50": 50, "sma200": 200}
    for name, p in periods.items():
        if len(closes) >= p:
            val = _ema(closes, p)[-1] if "ema" in name else _sma(closes, p)[-1]
            result[name] = {
                "value":     round(val, 2),
                "diff_pct":  round((price - val) / val * 100, 2),
                "above":     price > val,
            }
        else:
            result[name] = None

    # Golden / Death cross (50 vs 200)
    cross_signal = "NONE"
    cross_note   = ""
    if result.get("ema50") and result.get("sma200"):
        if result["ema50"]["above"] and not result.get("_prev_50_above", False):
            cross_signal = "GOLDEN_CROSS"
            cross_note   = "Price above both 50 & 200 MA — golden cross territory."
        elif not result["ema50"]["above"]:
            cross_signal = "DEATH_CROSS"
            cross_note   = "50 EMA below 200 SMA — death cross, long-term bearish."
        elif result["ema50"]["above"]:
            cross_signal = "BULLISH_ALIGNMENT"
            cross_note   = "50 EMA above 200 SMA — bullish long-term alignment."

    # Price vs MA stack
    above_count = sum(1 for k in ["ema20","ema50","sma200"] if result.get(k) and result[k]["above"])
    stack_signal = (
        "FULL_BULL" if above_count == 3
        else "PARTIAL_BULL" if above_count == 2
        else "PARTIAL_BEAR" if above_count == 1
        else "FULL_BEAR"
    )

    result["cross_signal"] = cross_signal
    result["cross_note"]   = cross_note
    result["stack_signal"] = stack_signal
    result["above_count"]  = above_count

    return result


# ─────────────────────────────────────────────────────────────────
# Bollinger Bands
# ─────────────────────────────────────────────────────────────────

def calc_bollinger(closes: List[float], period: int = 20, std_dev: float = 2.0) -> Dict[str, Any]:
    if len(closes) < period:
        return {"signal": "NEUTRAL", "note": "Insufficient data"}

    sma   = _sma(closes, period)[-1]
    std   = _stdev(closes[-period:])
    upper = round(sma + std_dev * std, 2)
    lower = round(sma - std_dev * std, 2)
    price = closes[-1]
    bw    = round((upper - lower) / sma * 100, 2)  # bandwidth %
    pctb  = round((price - lower) / (upper - lower) * 100, 1) if upper != lower else 50

    if price >= upper:
        signal = "UPPER_BAND"
        note   = "Price at upper Bollinger Band — overbought short-term, watch for reversal or breakout confirmation."
    elif price <= lower:
        signal = "LOWER_BAND"
        note   = "Price at lower Bollinger Band — oversold short-term, potential bounce setup."
    elif pctb > 70:
        signal = "UPPER_HALF"
        note   = "Price in upper half of Bollinger Bands — moderate bullish momentum."
    elif pctb < 30:
        signal = "LOWER_HALF"
        note   = "Price in lower half of Bollinger Bands — moderate bearish pressure."
    else:
        signal = "MID_BAND"
        note   = "Price mid-channel — no clear Bollinger signal."

    squeeze = bw < 5.0  # Low bandwidth = volatility squeeze = big move coming

    return {
        "upper":     upper,
        "middle":    round(sma, 2),
        "lower":     lower,
        "bandwidth": bw,
        "pct_b":     pctb,
        "signal":    signal,
        "squeeze":   squeeze,
        "note":      note + (" ⚡ SQUEEZE DETECTED — breakout may be imminent." if squeeze else ""),
    }


# ─────────────────────────────────────────────────────────────────
# Volume Surge
# ─────────────────────────────────────────────────────────────────

def calc_volume_surge(volumes: List[int], avg_period: int = 20) -> Dict[str, Any]:
    if len(volumes) < avg_period + 1:
        return {"signal": "NEUTRAL", "ratio": 1.0, "note": "Insufficient data"}

    avg_vol  = sum(volumes[-avg_period-1:-1]) / avg_period
    cur_vol  = volumes[-1]
    ratio    = round(cur_vol / avg_vol, 2) if avg_vol > 0 else 1.0

    if ratio >= 3.0:
        signal = "EXTREME_SURGE"
        note   = f"Volume {ratio}x average — extreme institutional activity or news catalyst."
    elif ratio >= 2.0:
        signal = "SURGE"
        note   = f"Volume {ratio}x average — significant surge, strong conviction behind the move."
    elif ratio >= 1.5:
        signal = "ELEVATED"
        note   = f"Volume {ratio}x average — above-average interest, trend may be strengthening."
    elif ratio <= 0.5:
        signal = "DRY"
        note   = f"Volume {ratio}x average — low participation, move may lack conviction."
    else:
        signal = "NORMAL"
        note   = f"Volume near average ({ratio}x) — no unusual activity."

    return {
        "current":    cur_vol,
        "average":    round(avg_vol),
        "ratio":      ratio,
        "signal":     signal,
        "note":       note,
        "is_surge":   ratio >= 1.5,
    }


# ─────────────────────────────────────────────────────────────────
# Channel Detection
# ─────────────────────────────────────────────────────────────────

def detect_channel(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    lookback: int = 40,
) -> Dict[str, Any]:
    """
    Detect price channels using linear regression on highs and lows.
    Identifies: trending channels (up/down), horizontal range, wedges.
    """
    if len(highs) < lookback:
        lookback = len(highs)

    h = highs[-lookback:]
    l = lows[-lookback:]
    c = closes[-lookback:]
    n = len(h)
    x = list(range(n))

    # Linear regression on highs and lows
    slope_h, intercept_h, r2_h = _linreg(x, h)
    slope_l, intercept_l, r2_l = _linreg(x, l)
    slope_c, intercept_c, r2_c = _linreg(x, c)

    current_price = c[-1]

    # Channel boundaries at the current bar
    channel_top    = round(slope_h * (n-1) + intercept_h, 2)
    channel_bottom = round(slope_l * (n-1) + intercept_l, 2)
    channel_mid    = round((channel_top + channel_bottom) / 2, 2)
    channel_width  = round((channel_top - channel_bottom) / channel_mid * 100, 2)

    # Normalize slopes to % per bar
    avg_price  = sum(c) / len(c)
    slope_h_pct = slope_h / avg_price * 100
    slope_l_pct = slope_l / avg_price * 100
    slope_c_pct = slope_c / avg_price * 100

    # Channel type
    is_parallel  = abs(slope_h_pct - slope_l_pct) < 0.03
    is_ascending  = slope_c_pct > 0.02
    is_descending = slope_c_pct < -0.02

    if is_parallel and is_ascending:
        channel_type = "ASCENDING_CHANNEL"
        channel_note = f"Ascending price channel — bullish trend structure. Top: ${channel_top:.2f}, Bottom: ${channel_bottom:.2f}."
    elif is_parallel and is_descending:
        channel_type = "DESCENDING_CHANNEL"
        channel_note = f"Descending price channel — bearish trend structure. Top: ${channel_top:.2f}, Bottom: ${channel_bottom:.2f}."
    elif is_parallel:
        channel_type = "HORIZONTAL_RANGE"
        channel_note = f"Horizontal trading range (${channel_bottom:.2f}–${channel_top:.2f}). Watch for breakout direction."
    elif slope_h_pct < 0 and slope_l_pct > 0:
        channel_type = "WEDGE_CONVERGING"
        channel_note = f"Converging wedge pattern — volatility compression, large move likely soon."
    elif slope_h_pct > 0 and slope_l_pct < 0:
        channel_type = "WEDGE_EXPANDING"
        channel_note = "Expanding wedge — increasing volatility, less reliable direction."
    else:
        channel_type = "UNDEFINED"
        channel_note = "No clear channel structure identified in the lookback window."

    # Price position within channel (0 = bottom, 100 = top)
    if channel_top != channel_bottom:
        position_pct = round((current_price - channel_bottom) / (channel_top - channel_bottom) * 100, 1)
    else:
        position_pct = 50

    # Near support/resistance?
    near_top    = position_pct >= 85
    near_bottom = position_pct <= 15
    support_note = (
        "⚠️  Price near channel TOP — potential resistance, wait for breakout or avoid calls."
        if near_top else
        "✅  Price near channel BOTTOM — potential support, favorable for calls / bounce plays."
        if near_bottom else
        f"Price at {position_pct:.0f}% of channel — mid-range, directional plays viable."
    )

    return {
        "type":           channel_type,
        "note":           channel_note,
        "support_note":   support_note,
        "top":            channel_top,
        "bottom":         channel_bottom,
        "mid":            channel_mid,
        "width_pct":      channel_width,
        "position_pct":   position_pct,
        "near_top":       near_top,
        "near_bottom":    near_bottom,
        "slope_pct":      round(slope_c_pct, 4),
        "r_squared":      round(r2_c, 3),
    }


# ─────────────────────────────────────────────────────────────────
# Directional Bias Score
# ─────────────────────────────────────────────────────────────────

def calc_bias_score(
    rsi: dict,
    macd: dict,
    mas: dict,
    volume: dict,
    channel: dict,
    bb: dict,
    current_price: float,
) -> Tuple[float, List[dict]]:
    """
    Aggregates all indicators into a single directional bias score.
    Returns (bias_score: -100 to +100, signals: list of signal dicts)
    """
    score   = 0.0
    signals = []

    def add(points: float, label: str, note: str, indicator: str):
        nonlocal score
        score += points
        direction = "BULLISH" if points > 0 else "BEARISH" if points < 0 else "NEUTRAL"
        signals.append({
            "indicator": indicator,
            "label":     label,
            "direction": direction,
            "points":    points,
            "note":      note,
        })

    # RSI (max ±20)
    rsi_v = rsi["value"]
    if rsi_v >= 70:
        add(-8,  "Overbought",     rsi["note"], "RSI")
    elif rsi_v <= 30:
        add(+8,  "Oversold",       rsi["note"], "RSI")
    elif 55 <= rsi_v < 70:
        add(+15, "Bullish Zone",   rsi["note"], "RSI")
    elif 30 < rsi_v <= 45:
        add(-15, "Bearish Zone",   rsi["note"], "RSI")
    else:
        add(0,   "Neutral",        rsi["note"], "RSI")

    # MACD (max ±20)
    if macd["signal"] in ("BULLISH_CROSS", "BULLISH"):
        add(+20, macd["signal"], macd["note"], "MACD")
    elif macd["signal"] in ("BEARISH_CROSS", "BEARISH"):
        add(-20, macd["signal"], macd["note"], "MACD")
    else:
        add(0, "Neutral", macd["note"], "MACD")

    # Moving Averages (max ±20)
    above = mas.get("above_count", 2)
    ma_pts = (above - 1.5) / 1.5 * 20   # maps 0→-20, 1.5→0, 3→+20
    add(round(ma_pts, 1), mas.get("stack_signal",""), mas.get("cross_note",""), "Moving Averages")

    # Volume (max ±10)
    if volume["signal"] in ("SURGE", "EXTREME_SURGE"):
        # Volume surge direction assumed to follow last close direction
        add(+8,  "Volume Surge",   volume["note"], "Volume")
    elif volume["signal"] == "DRY":
        add(-5,  "Low Volume",     volume["note"], "Volume")
    else:
        add(0,   volume["signal"], volume["note"], "Volume")

    # Channel (max ±15)
    ch = channel["type"]
    pos = channel.get("position_pct", 50)
    if ch == "ASCENDING_CHANNEL":
        add(+15, "Ascending Channel", channel["note"], "Channel")
    elif ch == "DESCENDING_CHANNEL":
        add(-15, "Descending Channel", channel["note"], "Channel")
    elif ch == "WEDGE_CONVERGING":
        add(+5,  "Converging Wedge", channel["note"], "Channel")
    elif ch == "HORIZONTAL_RANGE":
        # Near bottom of range = bullish setup
        ch_pts = round((50 - pos) / 50 * 10, 1)
        add(ch_pts, "Range", channel["support_note"], "Channel")
    else:
        add(0, ch, channel["note"], "Channel")

    # Bollinger (max ±10)
    if bb.get("squeeze"):
        add(+5, "BB Squeeze", bb["note"], "Bollinger")
    elif bb["signal"] == "LOWER_BAND":
        add(+8, "Lower Band Touch", bb["note"], "Bollinger")
    elif bb["signal"] == "UPPER_BAND":
        add(-8, "Upper Band Touch", bb["note"], "Bollinger")
    elif bb["signal"] == "UPPER_HALF":
        add(+4, "Upper Half", bb["note"], "Bollinger")
    elif bb["signal"] == "LOWER_HALF":
        add(-4, "Lower Half", bb["note"], "Bollinger")

    # Clamp
    final_score = round(max(-100, min(100, score)), 1)
    return final_score, signals


# ─────────────────────────────────────────────────────────────────
# Mock OHLCV data generator
# ─────────────────────────────────────────────────────────────────

def generate_mock_bars(symbol: str, num_bars: int = 200) -> List[Bar]:
    """
    Generates realistic OHLCV data with trending + mean-reverting behavior.
    Used in development before connecting to Schwab price history.
    """
    seed_prices = {
        "AAPL": 221.5,  "MSFT": 415.0,  "NVDA": 138.0,  "META": 632.0,
        "GOOGL": 192.0, "AMZN": 227.0,  "TSLA": 295.0,  "SPY": 590.0,
        "QQQ": 505.0,   "GLD": 231.0,   "AMD": 112.0,   "CRM": 325.0,
        "NFLX": 1015.0, "UBER": 82.0,   "COIN": 255.0,
    }

    # Use symbol hash as seed for reproducible but varied data per symbol
    rng_seed = sum(ord(c) for c in symbol)
    random.seed(rng_seed)

    price     = seed_prices.get(symbol, 100.0)
    daily_vol = 0.015 + random.uniform(0, 0.025)  # 1.5-4% daily vol
    trend     = random.uniform(-0.0005, 0.001)     # slight upward drift
    bars      = []

    from datetime import datetime, timedelta
    start_date = datetime.now() - timedelta(days=num_bars)

    for i in range(num_bars):
        # Geometric Brownian Motion
        ret    = trend + daily_vol * _randn()
        price *= (1 + ret)
        price  = max(price, 1.0)

        rng_h = abs(_randn()) * daily_vol * price
        rng_l = abs(_randn()) * daily_vol * price
        high  = round(price + rng_h, 2)
        low   = round(price - rng_l, 2)
        open_ = round(price * (1 + _randn() * 0.005), 2)
        vol   = int(random.gauss(10_000_000, 3_000_000))

        date_str = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
        bars.append(Bar(date=date_str, open=open_, high=high, low=low, close=round(price, 2), volume=max(vol, 100_000)))

    random.seed()  # reset seed
    return bars


# ─────────────────────────────────────────────────────────────────
# Math helpers
# ─────────────────────────────────────────────────────────────────

def _ema(values: List[float], period: int) -> List[float]:
    k      = 2 / (period + 1)
    result = [sum(values[:period]) / period]
    for v in values[period:]:
        result.append(v * k + result[-1] * (1 - k))
    return result


def _sma(values: List[float], period: int) -> List[float]:
    return [sum(values[i:i+period]) / period for i in range(len(values) - period + 1)]


def _stdev(values: List[float]) -> float:
    n   = len(values)
    if n < 2:
        return 0.0
    avg = sum(values) / n
    return math.sqrt(sum((x - avg)**2 for x in values) / (n - 1))


def _linreg(x: List[float], y: List[float]) -> Tuple[float, float, float]:
    """Returns (slope, intercept, r_squared)."""
    n   = len(x)
    if n < 2:
        return 0.0, y[0] if y else 0.0, 0.0
    sx  = sum(x); sy  = sum(y)
    sxy = sum(xi*yi for xi,yi in zip(x,y))
    sx2 = sum(xi**2 for xi in x)
    denom = n*sx2 - sx**2
    if denom == 0:
        return 0.0, sy/n, 0.0
    slope = (n*sxy - sx*sy) / denom
    intercept = (sy - slope*sx) / n

    # R²
    y_mean  = sy / n
    ss_tot  = sum((yi - y_mean)**2 for yi in y)
    y_pred  = [slope*xi + intercept for xi in x]
    ss_res  = sum((yi - yp)**2 for yi, yp in zip(y, y_pred))
    r2      = 1 - ss_res/ss_tot if ss_tot > 0 else 0.0

    return slope, intercept, r2


def _randn() -> float:
    """Box-Muller normal sample without numpy."""
    u1 = random.random()
    u2 = random.random()
    return math.sqrt(-2 * math.log(max(u1, 1e-10))) * math.cos(2 * math.pi * u2)
