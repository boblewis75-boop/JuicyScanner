"""
scanner.py
----------
JuicyScanner Core — Options Play Scanner

Takes raw options chain data + GEX data + flow data and:
  1. Flattens all contracts into a list
  2. Runs full technical analysis (RSI, StochRSI, MACD, MAs, ATR, OBV, VWAP, Candlesticks)
  3. Scores GEX confluence — is this strike near the GEX pin/wall?
  4. Scores flow confluence — is unusual flow confirming the direction?
  5. Filters by liquidity, spread, DTE
  6. Calculates R:R, target %, stop %
  7. Runs probability engine
  8. Multi-factor AI scoring with learned weights from play history
  9. Pattern boost — setups similar to past winners get a score bump
 10. Auto-saves all surfaced plays to play history for learning
 11. Returns top plays sorted by score
"""

import math
from typing import List, Dict, Any, Optional

from core.probability import calculate_likelihood
from core.technicals import analyze as analyze_technicals, generate_mock_bars
from core.brain import auto_log_scan, get_pattern_boost, get_weights, DEFAULT_WEIGHTS


def detect_regime(tech: Dict) -> str:
    bias   = tech.get("bias_score", 0)
    atr    = tech.get("atr", {})
    atr_r  = atr.get("regime", "NORMAL") if isinstance(atr, dict) else "NORMAL"
    if atr_r == "HIGH_VOL":
        return "high_vol"
    if bias >= 30:
        return "bull"
    if bias <= -30:
        return "bear"
    return "chop"


def score_gex_confluence(strike, option_type, gex_data):
    if not gex_data or not isinstance(gex_data, dict):
        return {"score": 0, "near_pin": False, "near_wall": False, "direction_ok": False, "note": "No GEX data"}

    spot      = gex_data.get("spot", 0)
    pin       = gex_data.get("pin_strike")
    call_wall = gex_data.get("call_wall")
    put_wall  = gex_data.get("put_wall")
    net_gex   = gex_data.get("net_gex", 0)
    rows      = gex_data.get("rows", [])

    if not spot or not pin:
        return {"score": 0, "near_pin": False, "near_wall": False, "direction_ok": False, "note": "Incomplete GEX data"}

    strike_gex = 0
    for row in rows:
        if abs(row.get("strike", -1) - strike) < 0.01:
            strike_gex = row.get("total", 0)
            break

    step  = spot * 0.005
    score = 0
    notes = []

    near_pin = pin and abs(strike - pin) <= step * 2
    if near_pin:
        score += 40
        notes.append(f"Strike ${strike} near GEX pin ${pin}")

    near_call_wall = call_wall and abs(strike - call_wall) <= step * 2
    if near_call_wall:
        score += 25 if option_type == "PUT" else 20
        notes.append(f"Near call wall ${call_wall}")

    near_put_wall = put_wall and abs(strike - put_wall) <= step * 2
    if near_put_wall:
        score += 25 if option_type == "CALL" else 20
        notes.append(f"Near put wall ${put_wall}")

    direction_ok = False
    if net_gex < 0:
        score += 20
        direction_ok = True
        notes.append(f"Negative GEX ({net_gex:.0f}B) — amplifies moves")
    elif net_gex > 1:
        score -= 10
        notes.append(f"Positive GEX ({net_gex:.0f}B) — may dampen move")
    else:
        direction_ok = True

    if abs(strike_gex) > 0.5:
        if (option_type == "CALL" and strike_gex > 0) or (option_type == "PUT" and strike_gex < 0):
            score += 15
            notes.append(f"GEX at strike: {strike_gex:.1f}B")

    score = max(0, min(100, score))
    return {
        "score": score, "near_pin": bool(near_pin),
        "near_wall": bool(near_call_wall or near_put_wall),
        "direction_ok": direction_ok, "net_gex": net_gex,
        "strike_gex": strike_gex, "pin_strike": pin,
        "note": "; ".join(notes) if notes else "No significant GEX confluence",
    }


def score_flow_confluence(symbol, option_type, strike, dte, flow_data):
    if not flow_data or not isinstance(flow_data, dict):
        return {"score": 0, "confirmed": False, "sweeps": 0, "note": "No flow data"}

    prints = flow_data.get("prints", [])
    sym_prints = [p for p in prints if p.get("symbol", "").upper() == symbol.upper()]
    if not sym_prints:
        return {"score": 0, "confirmed": False, "sweeps": 0, "note": f"No flow for {symbol}"}

    score = 0; notes = []; sweeps = 0; bull_flow = 0; bear_flow = 0

    for p in sym_prints:
        sentiment  = p.get("sentiment", "NEUTRAL")
        trade_type = p.get("trade_type", "")
        conviction = p.get("conviction_score", 0)
        if sentiment in ("BULLISH", "VERY_BULLISH"):
            bull_flow += conviction
        elif sentiment in ("BEARISH", "VERY_BEARISH"):
            bear_flow += conviction
        if trade_type in ("SWEEP", "BLOCK"):
            sweeps += 1

    total_flow = bull_flow + bear_flow
    confirmed  = False
    if total_flow > 0:
        flow_alignment = bull_flow / total_flow if option_type == "CALL" else bear_flow / total_flow
        if flow_alignment >= 0.70:
            score += 50; confirmed = True
            notes.append(f"{int(flow_alignment*100)}% flow aligns with {option_type}")
        elif flow_alignment >= 0.55:
            score += 25; notes.append(f"Moderate flow alignment ({int(flow_alignment*100)}%)")
        else:
            score -= 10; notes.append("Flow against direction")

    if sweeps >= 3:
        score += 25; notes.append(f"{sweeps} sweeps — institutional")
    elif sweeps >= 1:
        score += 10; notes.append(f"{sweeps} sweep detected")

    total_notional = sum(p.get("notional", 0) for p in sym_prints)
    if total_notional >= 5_000_000:
        score += 15; notes.append(f"${total_notional/1e6:.1f}M notional")
    elif total_notional >= 1_000_000:
        score += 8; notes.append(f"${total_notional/1e6:.1f}M notional")

    return {
        "score": max(0, min(100, score)), "confirmed": confirmed, "sweeps": sweeps,
        "bull_flow": round(bull_flow, 1), "bear_flow": round(bear_flow, 1),
        "notional": round(total_notional),
        "note": "; ".join(notes) if notes else "Flow neutral",
    }


def scan_options(
    chain: dict, quote: dict,
    min_rr: float = 1.5, min_target: float = 3.0, max_dte: int = 45,
    min_score: int = 40, min_oi: int = 50, max_spread_pct: float = 0.35,
    contract_type: str = "ALL", bars: list = None,
    gex_data: dict = None, flow_data: dict = None, auto_save: bool = True,
) -> List[Dict[str, Any]]:
    symbol           = chain["symbol"]
    underlying_price = chain["underlyingPrice"]

    if bars is None or len(bars) == 0:
        bars = generate_mock_bars(symbol)

    tech = analyze_technicals(bars, symbol=symbol)
    if tech:
        tech["current_price"] = underlying_price
    tech_bull_score = tech.get("bull_score", 50)
    tech_bias       = tech.get("bias_score", 0)
    regime          = detect_regime(tech)
    weights         = get_weights(regime)

    contracts = _flatten_chain(chain, contract_type)
    results   = []

    for c in contracts:
        if c.get("daysToExpiration", 0) > max_dte: continue
        oi_val  = c.get("openInterest", 0) or 0
        vol_val = c.get("totalVolume", 0) or 0
        effective_oi = oi_val if oi_val > 0 else vol_val
        if effective_oi < min_oi: continue
        if c.get("bid", 0) <= 0: continue
        if c.get("nonStandard", False): continue

        bid_p  = c.get("bid", 0) or 0
        ask_p  = c.get("ask", 0) or 0
        mid    = (bid_p + ask_p) / 2
        spread = (ask_p - bid_p) / mid if mid > 0 else 999
        c["bid"] = bid_p; c["ask"] = ask_p
        if spread > max_spread_pct: continue

        premium    = round(mid, 2)
        target_pct = _calc_target_pct(c, underlying_price)
        stop_pct   = _calc_stop_pct(premium)
        rr         = _calc_rr(target_pct, stop_pct,
                               delta=abs(c.get("delta", 0.4) or 0.4),
                               dte=c.get("daysToExpiration", 30) or 30)
        iv_rank    = _estimate_iv_rank(c.get("volatility", 50) or 50)
        breakeven  = _calc_breakeven(c, premium)

        if rr < min_rr: continue
        if abs(target_pct) > 80: continue

        put_call = c.get("putCall", "CALL")
        if put_call == "CALL" and tech_bias < -60: continue
        if put_call == "PUT" and tech_bias > 60: continue

        gex_conf  = score_gex_confluence(c["strikePrice"], put_call, gex_data)
        flow_conf = score_flow_confluence(symbol, put_call, c["strikePrice"],
                                          c.get("daysToExpiration", 30), flow_data)

        vol_oi_ratio = (vol_val / oi_val) if oi_val > 0 else 0
        prob = calculate_likelihood(
            option_type=put_call, underlying_price=underlying_price,
            strike=c["strikePrice"], premium=premium,
            iv=(c.get("volatility", 50) or 50) / 100,
            dte=c.get("daysToExpiration", 30) or 30,
            delta=abs(c.get("delta", 0.3) or 0.3),
            target_pct=abs(target_pct),
            tech_score=tech_bull_score if put_call == "CALL" else (100 - tech_bull_score),
        )

        score = _score_play(
            rr=rr, target_pct=target_pct, iv_rank=iv_rank,
            delta=abs(c.get("delta", 0.3) or 0.3),
            oi=effective_oi, volume=vol_val, vol_oi_ratio=vol_oi_ratio,
            dte=c.get("daysToExpiration", 30) or 30, spread_pct=spread,
            theta=c.get("theta", 0) or 0, contract_type=put_call,
            likelihood_pct=prob.get("likelihood_pct", 50),
            tech_bias=tech_bias,
            gex_score=gex_conf.get("score", 0),
            flow_score=flow_conf.get("score", 0),
            weights=weights,
        )

        if score < min_score: continue

        play = {
            "ticker": symbol, "type": put_call,
            "strike": c.get("strikePrice", 0),
            "expiry": _format_expiry(c.get("expirationDate", "")),
            "dte": c.get("daysToExpiration", 0), "premium": premium,
            "bid": c.get("bid", 0), "ask": c.get("ask", 0),
            "rr": round(rr, 1), "target": round(abs(target_pct), 1),
            "stop": round(stop_pct, 1), "breakeven": round(breakeven, 2),
            "underlyingPx": underlying_price, "likelihood": prob,
            "ivRank": iv_rank, "iv": round(c.get("volatility", 0) or 0, 1),
            "delta": round(c.get("delta", 0), 3),
            "theta": round(c.get("theta", 0), 4),
            "gamma": round(c.get("gamma", 0), 4),
            "vega": round(c.get("vega", 0), 4),
            "oi": effective_oi, "volume": vol_val, "score": score,
            "tech_bias": tech_bias, "regime": regime,
            "tech_signals": [s for s in tech.get("signals", []) if abs(s["points"]) >= 10],
            "gex_confluence": gex_conf, "flow_confluence": flow_conf,
        }

        boost, boost_reason = get_pattern_boost(play)
        play["score"]         = min(100, round(score + boost))
        play["pattern_boost"] = round(boost, 1)
        play["pattern_note"]  = boost_reason
        play["rationale"]     = _generate_rationale(
            symbol, put_call, target_pct, rr, iv_rank,
            c.get("delta", 0), play["score"], c.get("daysToExpiration", 0),
            prob["likelihood_pct"], prob["confidence"], tech_bias,
            gex_conf, flow_conf, boost_reason,
        )
        results.append(play)

    results.sort(key=lambda x: x["score"], reverse=True)
    output = results[:20]
    if output:
        output[0]["tech_full"] = tech

    if auto_save and output:
        try:
            auto_log_scan(output, {"regime": regime, "tech_bias": tech_bias,
                                    "gex_loaded": bool(gex_data), "flow_loaded": bool(flow_data)})
        except Exception as e:
            print(f"[Scanner] auto_save failed: {e}")

    return output


def _flatten_chain(chain: dict, contract_type: str) -> List[dict]:
    contracts = []
    def _extract(exp_map):
        for exp_date, options in exp_map.items():
            if isinstance(options, dict):
                for strike_str, contract_list in options.items():
                    if isinstance(contract_list, list):
                        for c in contract_list:
                            if isinstance(c, dict): contracts.append(c)
                    elif isinstance(contract_list, dict):
                        contracts.append(contract_list)
            elif isinstance(options, list):
                for opt in options:
                    if isinstance(opt, dict): contracts.append(opt)
    if contract_type in ("ALL", "CALL"): _extract(chain.get("callExpDateMap", {}))
    if contract_type in ("ALL", "PUT"):  _extract(chain.get("putExpDateMap", {}))
    return contracts


def _calc_target_pct(contract, underlying_price):
    premium = (contract["bid"] + contract["ask"]) / 2
    delta   = abs(contract.get("delta", 0) or 0)
    if delta < 0.01: return 999
    pct = (premium / delta / underlying_price) * 100
    return -pct if contract["putCall"] == "PUT" else pct

def _calc_stop_pct(premium): return -50.0

def _calc_rr(target_pct, stop_pct, delta=0.4, dte=30):
    target = abs(target_pct)
    if target <= 0 or delta <= 0: return 0
    if target < 5:    rr = 3.5
    elif target < 10: rr = 2.8
    elif target < 15: rr = 2.2
    elif target < 25: rr = 1.8
    elif target < 40: rr = 1.4
    else:             rr = 1.0
    if 0.35 <= delta <= 0.55:        rr *= 1.15
    elif delta > 0.70 or delta < 0.20: rr *= 0.85
    return round(rr, 2)

def _estimate_iv_rank(iv_pct):
    if iv_pct <= 15:   return 10
    elif iv_pct <= 25: return int(10 + (iv_pct-15)*2.5)
    elif iv_pct <= 40: return int(35 + (iv_pct-25)*1.3)
    elif iv_pct <= 60: return int(55 + (iv_pct-40)*1.0)
    elif iv_pct <= 100:return int(75 + (iv_pct-60)*0.5)
    else:              return min(99, int(95 + (iv_pct-100)*0.1))

def _calc_breakeven(contract, premium):
    return contract["strikePrice"] + premium if contract["putCall"] == "CALL" else contract["strikePrice"] - premium

def _format_expiry(date_str):
    try:
        from datetime import datetime
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
        return f"{dt.month}/{dt.day}"
    except: return date_str


def _score_play(rr, target_pct, iv_rank, delta, oi, volume, dte, spread_pct, theta,
                contract_type, likelihood_pct=50.0, tech_bias=0.0, vol_oi_ratio=0.0,
                gex_score=0.0, flow_score=0.0, weights=None):
    w = weights or DEFAULT_WEIGHTS

    def lkh_pts(lkh):
        if lkh >= 65: return 1.0
        if lkh >= 55: return 0.8
        if lkh >= 45: return 0.56
        if lkh >= 38: return 0.32
        return 0.08

    def tech_pts(bias, ctype):
        aligned = bias if ctype == "CALL" else -bias
        if aligned >= 40:  return 1.0
        if aligned >= 20:  return 0.75
        if aligned >= 0:   return 0.5
        if aligned >= -20: return 0.25
        return 0.0

    def rr_pts(rr):
        if rr >= 3.5: return 1.0
        if rr >= 2.8: return 0.8
        if rr >= 2.2: return 0.6
        if rr >= 1.8: return 0.4
        return 0.13

    def ivr_pts(ivr):
        if 20 <= ivr <= 50:  return 1.0
        if 15 <= ivr <= 60:  return 0.7
        if 10 <= ivr <= 70:  return 0.4
        return 0.1

    def delta_pts(d):
        if 0.35 <= d <= 0.50:  return 1.0
        if 0.28 <= d <= 0.58:  return 0.7
        if 0.22 <= d <= 0.65:  return 0.4
        return 0.1

    def dte_pts(dte):
        if 21 <= dte <= 45:  return 1.0
        if 14 <= dte <= 55:  return 0.7
        if 10 <= dte <= 60:  return 0.4
        if dte < 10:         return 0.1
        return 0.3

    def liq_pts(oi, vol, voi):
        base = min(1.0, (math.log10(max(oi, 1)) / math.log10(10000)))
        if voi >= 2.0:   base = min(1.0, base + 0.4)
        elif voi >= 0.5: base = min(1.0, base + 0.2)
        return base

    factor_values = {
        "likelihood": lkh_pts(likelihood_pct),
        "tech_bias":  tech_pts(tech_bias, contract_type),
        "gex_conf":   gex_score / 100.0,
        "flow_conf":  flow_score / 100.0,
        "rr_ratio":   rr_pts(rr),
        "iv_rank":    ivr_pts(iv_rank),
        "delta":      delta_pts(delta),
        "dte":        dte_pts(dte),
        "liquidity":  liq_pts(oi, volume, vol_oi_ratio),
    }

    if spread_pct <= 0.05:   spread_bonus = 5
    elif spread_pct <= 0.10: spread_bonus = 4
    elif spread_pct <= 0.15: spread_bonus = 3
    elif spread_pct <= 0.25: spread_bonus = 2
    else:                    spread_bonus = 0

    raw   = sum(factor_values.get(k, 0) * w.get(k, 0) for k in w)
    score = raw * 95 + spread_bonus
    return min(100, max(0, round(score)))


def _generate_rationale(symbol, opt_type, target_pct, rr, iv_rank, delta, score,
                         dte, likelihood_pct=50.0, confidence="MEDIUM", tech_bias=0.0,
                         gex_conf=None, flow_conf=None, pattern_note=""):
    t         = abs(target_pct)
    direction = "bullish" if opt_type == "CALL" else "bearish"
    iv_note   = ("Low IV — cheap entry." if iv_rank < 40 else
                 "High IV — consider spread." if iv_rank > 65 else "IV ideal.")
    tech_note = ("Tech confirms." if abs(tech_bias) >= 50 else
                 "Tech moderate." if abs(tech_bias) >= 20 else "Tech mixed.")
    parts = [f"{symbol} {direction}.", f"Needs {t:.1f}% move, {rr}:1 R:R.",
             f"Likelihood {likelihood_pct:.0f}% ({confidence}).", iv_note, tech_note]
    if isinstance(gex_conf, dict) and gex_conf.get("score", 0) > 20:
        parts.append(f"GEX: {gex_conf.get('note','')[:80]}")
    if isinstance(flow_conf, dict) and flow_conf.get("score", 0) > 20:
        parts.append(f"Flow: {flow_conf.get('note','')[:80]}")
    if pattern_note:
        parts.append(f"📈 {pattern_note}")
    return " ".join(parts)
