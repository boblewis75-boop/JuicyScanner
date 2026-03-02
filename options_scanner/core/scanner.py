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
    min_score: int        = 55,
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
    if bars is None:
        bars = generate_mock_bars(symbol)  # swap for real Schwab bars when live

    tech = analyze_technicals(bars, symbol=symbol)
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
        rr         = _calc_rr(target_pct, stop_pct)
        iv_rank    = _estimate_iv_rank(c.get("volatility", 50) or 50)
        breakeven  = _calc_breakeven(c, premium)

        # --- Filters on calculated values ---
        if rr < min_rr:
            continue
        if abs(target_pct) < min_target:
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
        score = _score_play(
            rr             = rr,
            target_pct     = target_pct,
            iv_rank        = iv_rank,
            delta          = abs(c["delta"]),
            oi             = c.get("openInterest", 0) or 0,
            volume         = c.get("totalVolume", 0) or 0,
            dte            = c.get("daysToExpiration", 30) or 30,
            spread_pct     = spread,
            theta          = c["theta"],
            contract_type  = c["putCall"],
            likelihood_pct = prob["likelihood_pct"],
            tech_bias      = tech_bias,
        )

        if score < min_score:
            continue

        results.append({
            # Core info
            "ticker":       symbol,
            "type":         c["putCall"],
            "strike":       c["strikePrice"],
            "expiry":       _format_expiry(c["expirationDate"]),
            "dte":          c["daysToExpiration"],
            "premium":      premium,
            "bid":          c["bid"],
            "ask":          c["ask"],
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
            "iv":           round(c["volatility"], 1),
            "delta":        round(c["delta"], 3),
            "theta":        round(c["theta"], 4),
            "gamma":        round(c["gamma"], 4),
            "vega":         round(c["vega"], 4),
            # Liquidity
            "oi":           c["openInterest"],
            "volume":       c["totalVolume"],
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


def _calc_rr(target_pct: float, stop_pct: float) -> float:
    """
    R:R = expected gain / max loss on the OPTION.
    Target 100% gain, stop at 50% loss = 2:1 base, modified by target distance.
    """
    target = abs(target_pct)
    if target <= 0:
        return 0
    rr_base = 2.0
    if target < 15:
        modifier = 1.8
    elif target < 25:
        modifier = 1.4
    elif target < 35:
        modifier = 1.1
    else:
        modifier = 0.8
    return round(rr_base * modifier, 2)


def _estimate_iv_rank(iv_pct: float) -> int:
    """Estimate IV rank (0-100) from raw IV percentage."""
    rank = min(99, max(1, int(iv_pct * 0.9 + random_jitter(iv_pct))))
    return rank


def random_jitter(val: float) -> float:
    return (math.sin(val * 7.3) * 10)


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
) -> int:
    """
    Multi-factor scoring model.

    Factors:
      R:R ratio           (20 pts)
      Target move size    (15 pts)
      IV Rank             (15 pts)
      Delta               (10 pts)
      Liquidity (OI)      (10 pts)
      DTE                 (10 pts)
      Play Likelihood %   (15 pts)  ← NEW
      Technical Bias      (5  pts)  ← NEW
    """
    score = 0.0

    # 1. R:R Ratio (20 pts)
    if rr >= 4.0:   score += 20
    elif rr >= 3.0: score += 16
    elif rr >= 2.5: score += 12
    elif rr >= 2.0: score += 8
    else:           score += 4

    # 2. Target move needed (15 pts) — lower is better
    t = abs(target_pct)
    if t < 10:      score += 15
    elif t < 18:    score += 12
    elif t < 25:    score += 8
    elif t < 35:    score += 4
    else:           score += 1

    # 3. IV Rank (15 pts)
    if 25 <= iv_rank <= 55:   score += 15
    elif 20 <= iv_rank <= 65: score += 10
    elif 15 <= iv_rank <= 75: score += 6
    else:                     score += 2

    # 4. Delta (10 pts)
    if 0.30 <= delta <= 0.50:   score += 10
    elif 0.25 <= delta <= 0.60: score += 7
    elif 0.20 <= delta <= 0.70: score += 4
    else:                       score += 1

    # 5. Liquidity (10 pts)
    liq_score = min(10, (math.log10(max(oi, 1)) / math.log10(50000)) * 10)
    score += liq_score

    # 6. DTE (10 pts)
    if 21 <= dte <= 35:   score += 10
    elif 14 <= dte <= 45: score += 7
    elif dte <= 14:       score += 3
    else:                 score += 5

    # 7. Play Likelihood % (15 pts) — NEW
    if likelihood_pct >= 70:   score += 15
    elif likelihood_pct >= 60: score += 12
    elif likelihood_pct >= 50: score += 8
    elif likelihood_pct >= 40: score += 4
    else:                      score += 1

    # 8. Technical Bias (5 pts) — NEW
    # Map -100..+100 bias to 0..5 pts for calls, 5..0 for puts
    if contract_type == "CALL":
        tech_pts = max(0, (tech_bias + 100) / 200 * 5)
    else:
        tech_pts = max(0, (100 - tech_bias) / 200 * 5)
    score += tech_pts

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
