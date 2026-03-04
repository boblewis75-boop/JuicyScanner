"""
scanner.py
----------
The brain of OptionsEdge AI.

Takes raw options chain data (from Schwab live OR mock) and:
  1. Flattens all contracts into a list
  2. Runs technical analysis on OHLCV bars
  3. Filters by liquidity, spread, DTE
  4. Calculates R:R ratio, target %, stop %
  5. Calculates play likelihood % using probability engine
  6. Scores each play 0-100 using multi-factor AI scoring
  7. Returns top plays sorted by score
"""

import math
from typing import List, Dict, Any, Optional

from core.probability import calculate_likelihood
from core.technicals import analyze as analyze_technicals, generate_mock_bars


# ----------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------

def scan_options(
    chain: dict,
    quote: dict,
    min_rr: float         = 1.5,
    min_target: float     = 3.0,
    max_dte: int          = 45,
    min_score: int        = 40,
    min_oi: int           = 50,
    max_spread_pct: float = 0.35,
    contract_type: str    = "ALL",   # "ALL", "CALL", "PUT"
    bars: list            = None,    # OHLCV bars for technical analysis (optional)
) -> List[Dict[str, Any]]:
    """
    Main scanner function.
    Returns a list of qualifying options plays, sorted by AI score descending.
    """
    symbol           = chain["symbol"]
    underlying_price = chain["underlyingPrice"]

    # ── Technical analysis (run once per symbol, not per contract) ──
    if bars is None or len(bars) == 0:
        bars = generate_mock_bars(symbol)

    tech = analyze_technicals(bars, symbol=symbol)
    # Override current_price with actual underlying price from chain
    if tech:
        tech["current_price"] = underlying_price
    tech_bull_score = tech.get("bull_score", 50)
    tech_bias       = tech.get("bias_score", 0)

    contracts = _flatten_chain(chain, contract_type)
    results   = []

    for c in contracts:
        # --- Basic filters ---
        if c.get("daysToExpiration", 0) > max_dte:
            continue
        if c.get("openInterest", 0) < min_oi:
            continue
        if c.get("bid", 0) <= 0:
            continue
        # Skip non-standard contracts
        if c.get("nonStandard", False):
            continue

        # Spread quality check
        bid_p  = c.get("bid", 0) or 0
        ask_p  = c.get("ask", 0) or 0
        mid    = (bid_p + ask_p) / 2
        spread = (ask_p - bid_p) / mid if mid > 0 else 999
        c["bid"] = bid_p
        c["ask"] = ask_p
        if spread > max_spread_pct:
            continue

        # --- Core calculations ---
        premium    = round(mid, 2)
        target_pct = _calc_target_pct(c, underlying_price)
        stop_pct   = _calc_stop_pct(premium)
        rr         = _calc_rr(target_pct, stop_pct, delta=abs(c.get('delta',0.4) or 0.4), dte=c.get('daysToExpiration',30) or 30)
        iv_rank    = _estimate_iv_rank(c.get("volatility", 50) or 50)
        breakeven  = _calc_breakeven(c, premium)

        # --- Filters on calculated values ---
        if rr < min_rr:
            continue
        # Filter out contracts that need HUGE moves (>80%) — too far OTM
        if abs(target_pct) > 80:
            continue
        # Alignment filter — skip calls in strong bear, puts in strong bull
        # Only enforce when tech signal is very strong (>60 pts either way)
        put_call = c.get("putCall", "CALL")
        if put_call == "CALL" and tech_bias < -60:
            continue
        if put_call == "PUT" and tech_bias > 60:
            continue

        # --- Play likelihood (probability engine) ---
        prob = calculate_likelihood(
            option_type      = c.get("putCall", "CALL"),
            underlying_price = underlying_price,
            strike           = c["strikePrice"],
            premium          = premium,
            iv               = (c.get("volatility", 50) or 50) / 100,
            dte              = c.get("daysToExpiration", 30) or 30,
            delta            = abs(c.get("delta", 0.3) or 0.3),
            target_pct       = abs(target_pct),
            tech_score       = tech_bull_score if c.get("putCall","CALL") == "CALL" else (100 - tech_bull_score),
        )

        # --- AI Scoring (now includes tech + likelihood) ---
        oi_val  = c.get("openInterest", 0) or 0
        vol_val = c.get("totalVolume", 0) or 0
        vol_oi_ratio = (vol_val / oi_val) if oi_val > 0 else 0
        score = _score_play(
            rr             = rr,
            target_pct     = target_pct,
            iv_rank        = iv_rank,
            delta          = abs(c.get("delta", 0.3) or 0.3),
            oi             = oi_val,
            volume         = vol_val,
            vol_oi_ratio   = vol_oi_ratio,
            dte            = c.get("daysToExpiration", 30) or 30,
            spread_pct     = spread,
            theta          = c.get("theta", 0) or 0,
            contract_type  = c.get("putCall", "CALL"),
            likelihood_pct = prob.get("likelihood_pct", 50),
            tech_bias      = tech_bias,
        )

        if score < min_score:
            continue

        results.append({
            # Core info
            "ticker":       symbol,
            "type":         c.get("putCall", "CALL"),
            "strike":       c.get("strikePrice", 0),
            "expiry":       _format_expiry(c.get("expirationDate", "")),
            "dte":          c.get("daysToExpiration", 0),
            "premium":      premium,
            "bid":          c.get("bid", 0),
            "ask":          c.get("ask", 0),
            # Risk metrics
            "rr":           round(rr, 1),
            "target":       round(abs(target_pct), 1),
            "stop":         round(stop_pct, 1),
            "breakeven":    round(breakeven, 2),
            "underlyingPx": underlying_price,
            # Probability
            "likelihood":   prob,
            # Greeks
            "ivRank":       iv_rank,
            "iv":           round(c.get("volatility", 0) or 0, 1),
            "delta":        round(c["delta"], 3),
            "theta":        round(c["theta"], 4),
            "gamma":        round(c["gamma"], 4),
            "vega":         round(c["vega"], 4),
            # Liquidity
            "oi":           c.get("openInterest", 0),
            "volume":       c.get("totalVolume", 0),
            # Scores
            "score":        score,
            # Technicals (summary — full tech object attached per-symbol below)
            "tech_bias":    tech_bias,
            "tech_signals": [s for s in tech.get("signals", []) if abs(s["points"]) >= 10],
            "rationale":    _generate_rationale(
                symbol, c["putCall"], target_pct, rr, iv_rank,
                c["delta"], score, c["daysToExpiration"],
                prob["likelihood_pct"], prob["confidence"], tech_bias
            ),
        })

    # Sort by score descending, take top 20
    results.sort(key=lambda x: x["score"], reverse=True)

    # Attach full tech analysis to the top result (avoid bloating every row)
    output = results[:20]
    if output:
        output[0]["tech_full"] = tech

    return output


# ----------------------------------------------------------------
# Calculation helpers
# ----------------------------------------------------------------

def _flatten_chain(chain: dict, contract_type: str) -> List[dict]:
    """
    Turn the nested Schwab chain structure into a flat list of contracts.
    
    Schwab live API format:
      callExpDateMap: { "2025-01-17:30": { "150.0": [contract], "155.0": [contract] } }
    
    Mock format:
      callExpDateMap: { "2025-01-17": [contract, contract, ...] }
    
    This function handles both.
    """
    contracts = []

    def _extract(exp_map):
        for exp_date, options in exp_map.items():
            # Real Schwab: options is a dict of {strike_str: [contract]}
            if isinstance(options, dict):
                for strike_str, contract_list in options.items():
                    if isinstance(contract_list, list):
                        for c in contract_list:
                            if isinstance(c, dict):
                                contracts.append(c)
                    elif isinstance(contract_list, dict):
                        contracts.append(contract_list)
            # Mock format: options is a list of contracts
            elif isinstance(options, list):
                for opt in options:
                    if isinstance(opt, dict):
                        contracts.append(opt)

    if contract_type in ("ALL", "CALL"):
        _extract(chain.get("callExpDateMap", {}))
    if contract_type in ("ALL", "PUT"):
        _extract(chain.get("putExpDateMap", {}))

    return contracts


def _calc_target_pct(contract: dict, underlying_price: float) -> float:
    """
    Target % = how far the underlying needs to move for a 100% gain on the option.
    Based on delta-adjusted move needed.
    """
    strike  = contract["strikePrice"]
    premium = (contract["bid"] + contract["ask"]) / 2
    delta   = abs(contract.get("delta", 0) or 0)

    if delta < 0.01:
        return 999
    move_needed = premium / delta
    pct_needed  = (move_needed / underlying_price) * 100

    if contract["putCall"] == "PUT":
        return -pct_needed
    return pct_needed


def _calc_stop_pct(premium: float) -> float:
    """Stop loss = -50% of premium paid."""
    return -50.0


def _calc_rr(target_pct: float, stop_pct: float, delta: float = 0.4, dte: int = 30) -> float:
    """
    Real R:R: if underlying moves target%, option gains ~(target_pct/delta)*delta*multiplier.
    Win = ~100% gain on option premium (target hit).
    Loss = -50% stop (standard debit stop).
    Penalize for time decay (theta burn on longer DTE low-delta plays).
    """
    target = abs(target_pct)
    if target <= 0 or delta <= 0:
        return 0
    # Reward plays where target is achievable — tighter target = better RR
    if target < 5:    rr = 3.5
    elif target < 10: rr = 2.8
    elif target < 15: rr = 2.2
    elif target < 25: rr = 1.8
    elif target < 40: rr = 1.4
    else:             rr = 1.0
    # Delta sweet spot bonus
    if 0.35 <= delta <= 0.55:
        rr *= 1.15
    elif delta > 0.70 or delta < 0.20:
        rr *= 0.85
    return round(rr, 2)


def _estimate_iv_rank(iv_pct: float) -> int:
    """
    Estimate IV rank (0-100) from raw IV percentage.
    Uses typical IV range per volatility bucket rather than random jitter.
    Stocks typically have IV between 15% (calm SPY) and 200% (biotech).
    """
    # Low IV = high rank if we want to buy premium? No — for debit plays:
    # Low IV = cheap premium = good. We want IV rank 20-50 for debit plays.
    # Map IV% to a rank assuming:
    #   15% IV = rank ~10 (very low, SPY-like)
    #   30% IV = rank ~35
    #   50% IV = rank ~55
    #   80% IV = rank ~75
    #   120%+ IV = rank ~90+
    if iv_pct <= 15:   return 10
    elif iv_pct <= 25: return int(10 + (iv_pct - 15) * 2.5)
    elif iv_pct <= 40: return int(35 + (iv_pct - 25) * 1.3)
    elif iv_pct <= 60: return int(55 + (iv_pct - 40) * 1.0)
    elif iv_pct <= 100:return int(75 + (iv_pct - 60) * 0.5)
    else:              return min(99, int(95 + (iv_pct - 100) * 0.1))


def _calc_breakeven(contract: dict, premium: float) -> float:
    if contract["putCall"] == "CALL":
        return contract["strikePrice"] + premium
    else:
        return contract["strikePrice"] - premium


def _format_expiry(date_str: str) -> str:
    try:
        from datetime import datetime
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
        return f"{dt.month}/{dt.day}"
    except Exception:
        return date_str


# ----------------------------------------------------------------
# AI Scoring Engine (0-100) — now includes likelihood + technicals
# ----------------------------------------------------------------

def _score_play(
    rr: float,
    target_pct: float,
    iv_rank: int,
    delta: float,
    oi: int,
    volume: int,
    dte: int,
    spread_pct: float,
    theta: float,
    contract_type: str,
    likelihood_pct: float = 50.0,
    tech_bias: float      = 0.0,
    vol_oi_ratio: float   = 0.0,
) -> int:
    """
    Multi-factor scoring model.

    Factors:
      Play Likelihood %   (25 pts)  — primary signal
      Technical Alignment (20 pts)  — trend confirmation
      R:R ratio           (15 pts)  — reward vs risk
      Delta sweet spot    (10 pts)  — option positioning
      IV Rank             (10 pts)  — entry pricing
      DTE window          (10 pts)  — time management
      Liquidity           ( 5 pts)  — OI + Vol/OI ratio
      Spread quality      ( 5 pts)  — slippage cost
    """
    score = 0.0

    # 1. Play Likelihood % (25 pts) — most important signal
    if likelihood_pct >= 65:   score += 25
    elif likelihood_pct >= 55: score += 20
    elif likelihood_pct >= 45: score += 14
    elif likelihood_pct >= 38: score += 8
    else:                      score += 2

    # 2. Technical Alignment (20 pts) — direction agreement
    if contract_type == "CALL":
        bias_aligned = tech_bias
    else:
        bias_aligned = -tech_bias  # for puts, negative bias is good
    if bias_aligned >= 40:    score += 20
    elif bias_aligned >= 20:  score += 15
    elif bias_aligned >= 0:   score += 10
    elif bias_aligned >= -20: score += 5
    else:                     score += 0

    # 3. R:R ratio (15 pts)
    if rr >= 3.5:   score += 15
    elif rr >= 2.8: score += 12
    elif rr >= 2.2: score += 9
    elif rr >= 1.8: score += 6
    else:           score += 2

    # 4. Delta sweet spot (10 pts) — 0.30-0.55 is ideal
    if 0.35 <= delta <= 0.50:   score += 10
    elif 0.28 <= delta <= 0.58: score += 7
    elif 0.22 <= delta <= 0.65: score += 4
    else:                       score += 1

    # 5. IV Rank (10 pts) — 20-50 ideal for debit buys
    if 20 <= iv_rank <= 50:   score += 10
    elif 15 <= iv_rank <= 60: score += 7
    elif 10 <= iv_rank <= 70: score += 4
    else:                     score += 1

    # 6. DTE window (10 pts) — 21-45 DTE is the sweet spot
    if 21 <= dte <= 45:   score += 10
    elif 14 <= dte <= 55: score += 7
    elif 10 <= dte <= 60: score += 4
    elif dte < 10:        score += 1
    else:                 score += 3

    # 7. Liquidity (5 pts) — OI + unusual Vol/OI activity
    liq_pts = min(3, (math.log10(max(oi, 1)) / math.log10(10000)) * 3)
    # Bonus for vol/OI > 0.5 (unusual activity signal)
    if vol_oi_ratio >= 2.0:   liq_pts = min(5, liq_pts + 2)
    elif vol_oi_ratio >= 0.5: liq_pts = min(5, liq_pts + 1)
    score += liq_pts

    # 8. Spread quality (5 pts) — tighter spread = better fill
    if spread_pct <= 0.05:   score += 5
    elif spread_pct <= 0.10: score += 4
    elif spread_pct <= 0.15: score += 3
    elif spread_pct <= 0.25: score += 2
    else:                    score += 0

    return min(100, max(0, round(score)))


# ----------------------------------------------------------------
# Rationale generator
# ----------------------------------------------------------------

def _generate_rationale(
    symbol: str,
    opt_type: str,
    target_pct: float,
    rr: float,
    iv_rank: int,
    delta: float,
    score: int,
    dte: int,
    likelihood_pct: float = 50.0,
    confidence: str       = "MEDIUM",
    tech_bias: float      = 0.0,
) -> str:
    t = abs(target_pct)
    direction = "bullish" if opt_type == "CALL" else "bearish"

    iv_note = (
        "Low IV rank — attractively priced debit play."
        if iv_rank < 40
        else "Elevated IV rank — consider a spread to reduce premium cost."
        if iv_rank > 65
        else "IV rank in the ideal zone for a debit play."
    )

    likelihood_note = (
        f"Play likelihood is {likelihood_pct:.0f}% ({confidence} confidence)."
    )

    tech_note = (
        "Technical analysis strongly supports the direction."
        if abs(tech_bias) >= 50
        else "Technicals are mixed — monitor closely."
        if abs(tech_bias) < 20
        else "Technicals offer moderate confirmation."
    )

    return (
        f"{symbol} {direction} setup. "
        f"Underlying needs a {t:.1f}% move with {rr}:1 R:R. "
        f"{likelihood_note} "
        f"{iv_note} "
        f"{tech_note}"
    )
