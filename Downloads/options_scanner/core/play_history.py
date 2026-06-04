"""
play_history.py
---------------
JuicyScanner Play History & Self-Learning Engine

Auto-saves every play the scanner surfaces.
Tracks outcomes when you close plays.
Learns which factor combinations actually win — no Claude API needed.

Learning method:
  - Exponential Moving Average weight adjustment per factor
  - Bayesian win-rate tracking per indicator combo
  - Regime detection (bull/bear/chop/highvol) — learns separately per regime
  - Pattern scoring: setups similar to historical winners get a boost
  - All insights generated from pure statistics on your own trade history
"""

import json
import os
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

# ── Storage ──────────────────────────────────────────────────────
_LOCAL = os.path.join(os.path.dirname(__file__), "..", "data", "play_history.json")
HISTORY_FILE = "/app/play_history.json" if os.path.exists("/app") else _LOCAL

# ── Default factor weights (sum to 1.0) ──────────────────────────
DEFAULT_WEIGHTS = {
    "likelihood":   0.20,
    "tech_bias":    0.15,
    "gex_conf":     0.12,   # GEX confluence
    "flow_conf":    0.12,   # Flow confluence
    "rr_ratio":     0.12,
    "iv_rank":      0.10,
    "delta":        0.08,
    "dte":          0.06,
    "liquidity":    0.05,
}

LEARNING_RATE = 0.04   # how fast weights shift per outcome
MIN_WEIGHT    = 0.02
MAX_WEIGHT    = 0.40


# ─────────────────────────────────────────────────────────────────
# Storage
# ─────────────────────────────────────────────────────────────────

def _load() -> Dict:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return _default_state()


def _save(data: Dict):
    try:
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        with open(HISTORY_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[PlayHistory] save error: {e}")


def _now() -> str:
    return datetime.now().isoformat()


def _default_state() -> Dict:
    return {
        "version":    "2.0",
        "created_at": _now(),
        "weights":    DEFAULT_WEIGHTS.copy(),
        "regime_weights": {
            "bull":     DEFAULT_WEIGHTS.copy(),
            "bear":     DEFAULT_WEIGHTS.copy(),
            "chop":     DEFAULT_WEIGHTS.copy(),
            "high_vol": DEFAULT_WEIGHTS.copy(),
        },
        "open_plays":   [],    # plays currently tracked (not yet closed)
        "closed_plays": [],    # plays with outcomes — the training data
        "stats": {
            "total_surfaced": 0,   # every play scanner ever showed
            "total_tracked":  0,   # plays you chose to track
            "total_closed":   0,
            "wins": 0, "losses": 0, "partials": 0, "expired": 0,
            "win_rate": 0.0,
            "avg_return_pct": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
            "profit_factor": 0.0,  # gross wins / gross losses
        },
        "factor_stats": {},     # win rate per factor bucket
        "combo_stats":  {},     # win rate per indicator combo
        "regime_stats": {},     # win rate per market regime
        "insights":     [],
    }


# ─────────────────────────────────────────────────────────────────
# Auto-save scan results (called every time scanner runs)
# ─────────────────────────────────────────────────────────────────

def auto_save_scan(plays: List[Dict], scan_context: Dict = None) -> int:
    """
    Called automatically after every scan.
    Saves all surfaced plays to history (status=SURFACED, not tracked).
    Returns count saved.
    """
    if not plays:
        return 0

    data    = _load()
    saved   = 0
    context = scan_context or {}

    for play in plays:
        play_id = _make_id(play)

        # Don't duplicate — skip if already logged this exact play today
        today = _now()[:10]
        exists = any(
            p["id"] == play_id and p.get("surfaced_date", "")[:10] == today
            for p in data["open_plays"] + data["closed_plays"]
        )
        if exists:
            continue

        record = {
            "id":            play_id,
            "status":        "SURFACED",   # not yet tracked by user
            "surfaced_date": _now(),

            # Identity
            "ticker":        play.get("ticker", ""),
            "type":          play.get("type", "CALL"),
            "strike":        play.get("strike", 0),
            "expiry":        play.get("expiry", ""),
            "dte":           play.get("dte", 0),
            "premium":       play.get("premium", 0),
            "underlying_px": play.get("underlyingPx", 0),

            # Scores
            "score":         play.get("score", 0),
            "likelihood_pct":play.get("likelihood", {}).get("likelihood_pct", 50),

            # Factors used at scan time
            "rr":            play.get("rr", 0),
            "target_pct":    play.get("target", 0),
            "iv_rank":       play.get("ivRank", 50),
            "delta":         play.get("delta", 0.3),
            "tech_bias":     play.get("tech_bias", 0),
            "gex_confluence":play.get("gex_confluence", {}),
            "flow_confluence":play.get("flow_confluence", {}),
            "tech_signals":  [s.get("label","") for s in play.get("tech_signals", [])],

            # Context
            "regime":        context.get("regime", "unknown"),
            "scan_context":  context,

            # Outcome (filled later)
            "tracked":       False,
            "entry_premium": None,
            "exit_premium":  None,
            "result":        None,
            "return_pct":    None,
            "closed_at":     None,
            "notes":         "",
        }

        data["open_plays"].append(record)
        data["stats"]["total_surfaced"] += 1
        saved += 1

    _save(data)
    return saved


# ─────────────────────────────────────────────────────────────────
# Mark a play as tracked (user decided to enter it)
# ─────────────────────────────────────────────────────────────────

def track_play(play_id: str, entry_premium: float = None, notes: str = "") -> Dict:
    """Mark a surfaced play as being tracked (you entered the trade)."""
    data = _load()
    play = next((p for p in data["open_plays"] if p["id"] == play_id), None)
    if not play:
        return {"success": False, "error": f"Play {play_id} not found in open plays"}

    play["tracked"]       = True
    play["entry_premium"] = entry_premium or play["premium"]
    play["notes"]         = notes
    play["tracked_at"]    = _now()
    data["stats"]["total_tracked"] += 1

    _save(data)
    return {"success": True, "play_id": play_id, "message": f"Now tracking {play['ticker']} {play['type']} ${play['strike']}"}


# ─────────────────────────────────────────────────────────────────
# Close a play with outcome → triggers learning
# ─────────────────────────────────────────────────────────────────

def close_play(
    play_id: str,
    exit_premium: float,
    result: str,      # WIN / PARTIAL / LOSS / EXPIRED
    notes: str = "",
) -> Dict:
    """
    Close a tracked play. This is the learning trigger — weights adjust based on outcome.
    result: WIN=100%+ gain, PARTIAL=1-99% gain, LOSS=-50% stop, EXPIRED=worthless
    """
    data = _load()
    play = next((p for p in data["open_plays"] if p["id"] == play_id), None)
    if not play:
        return {"success": False, "error": f"Play {play_id} not found"}

    entry     = play.get("entry_premium") or play["premium"]
    ret_pct   = round((exit_premium - entry) / entry * 100, 2) if entry > 0 else 0
    won       = result in ("WIN", "PARTIAL")

    # Finalize play record
    play.update({
        "status":       "CLOSED",
        "exit_premium": exit_premium,
        "return_pct":   ret_pct,
        "result":       result,
        "closed_at":    _now(),
        "close_notes":  notes,
    })

    # Move to closed
    data["open_plays"]   = [p for p in data["open_plays"] if p["id"] != play_id]
    data["closed_plays"].append(play)

    # Update aggregate stats
    s = data["stats"]
    s["total_closed"] += 1
    if result == "WIN":       s["wins"]     += 1
    elif result == "PARTIAL": s["partials"] += 1
    elif result == "EXPIRED": s["expired"]  += 1
    else:                     s["losses"]   += 1

    total = s["total_closed"]
    s["win_rate"] = round((s["wins"] + s["partials"] * 0.5) / total * 100, 1) if total > 0 else 0

    closed = data["closed_plays"]
    all_returns = [p["return_pct"] for p in closed if p.get("return_pct") is not None]
    s["avg_return_pct"] = round(sum(all_returns) / len(all_returns), 1) if all_returns else 0

    wins_ret   = [r for r in all_returns if r > 0]
    losses_ret = [r for r in all_returns if r <= 0]
    s["avg_win_pct"]  = round(sum(wins_ret)   / len(wins_ret),   1) if wins_ret   else 0
    s["avg_loss_pct"] = round(sum(losses_ret) / len(losses_ret), 1) if losses_ret else 0

    gross_wins   = sum(wins_ret)   if wins_ret   else 0
    gross_losses = abs(sum(losses_ret)) if losses_ret else 0
    s["profit_factor"] = round(gross_wins / gross_losses, 2) if gross_losses > 0 else 0

    # ── Learning: adjust weights ──────────────────────────────────
    regime = play.get("regime", "unknown")
    data["weights"] = _adjust_weights(data["weights"], play, won, ret_pct)
    if regime in data["regime_weights"]:
        data["regime_weights"][regime] = _adjust_weights(
            data["regime_weights"][regime], play, won, ret_pct
        )

    # ── Update factor + combo stats ───────────────────────────────
    _update_factor_stats(data, play, won, ret_pct)
    _update_combo_stats(data, play, won)

    # ── Regenerate insights every 5 closed plays ──────────────────
    if total >= 5 and total % 5 == 0:
        data["insights"] = _generate_insights(data)

    _save(data)
    return {
        "success":     True,
        "play_id":     play_id,
        "result":      result,
        "return_pct":  ret_pct,
        "new_weights": data["weights"],
        "message":     f"Closed {play['ticker']} {play['type']} ({result}, {ret_pct:+.1f}%). Weights updated.",
    }


# ─────────────────────────────────────────────────────────────────
# Weight adjustment engine
# ─────────────────────────────────────────────────────────────────

def _adjust_weights(weights: Dict, play: Dict, won: bool, return_pct: float) -> Dict:
    """
    EMA-based weight adjustment.
    Factors that were strong in winning plays get more weight.
    Factors that were strong in losing plays get less weight.
    Magnitude scales with how big the win/loss was.
    """
    magnitude = min(2.0, abs(return_pct) / 50.0)
    lr        = LEARNING_RATE * magnitude
    sign      = 1.0 if won else -1.0

    gex = play.get("gex_confluence", {})
    flow = play.get("flow_confluence", {})

    factor_scores = {
        "likelihood":  (play.get("likelihood_pct", 50)) / 100.0,
        "tech_bias":   abs(play.get("tech_bias", 0)) / 100.0,
        "gex_conf":    min(1.0, gex.get("score", 0) / 100.0) if isinstance(gex, dict) else 0.5,
        "flow_conf":   min(1.0, flow.get("score", 0) / 100.0) if isinstance(flow, dict) else 0.5,
        "rr_ratio":    min(1.0, play.get("rr", 2) / 5.0),
        "iv_rank":     1.0 - abs((play.get("iv_rank", 40) - 35)) / 65.0,
        "delta":       1.0 - abs((play.get("delta", 0.4) - 0.4)) / 0.5,
        "dte":         1.0 - abs((play.get("dte", 28) - 28)) / 45.0,
        "liquidity":   0.6,
    }

    new_weights = {}
    for factor, w in weights.items():
        fs    = factor_scores.get(factor, 0.5)
        adj   = lr * sign * (fs - 0.5)
        new_w = max(MIN_WEIGHT, min(MAX_WEIGHT, w + adj))
        new_weights[factor] = round(new_w, 4)

    # Renormalize
    total = sum(new_weights.values())
    return {k: round(v / total, 4) for k, v in new_weights.items()}


# ─────────────────────────────────────────────────────────────────
# Factor stats — win rate per factor bucket
# ─────────────────────────────────────────────────────────────────

def _update_factor_stats(data: Dict, play: Dict, won: bool, ret_pct: float):
    fs = data.setdefault("factor_stats", {})

    def _record(key: str, value):
        bucket = f"{key}:{value}"
        if bucket not in fs:
            fs[bucket] = {"wins": 0, "total": 0, "total_return": 0.0}
        fs[bucket]["total"] += 1
        fs[bucket]["total_return"] += ret_pct
        if won:
            fs[bucket]["wins"] += 1

    # Score bucket
    score = play.get("score", 0)
    _record("score_bucket", f"{(score // 10) * 10}-{(score // 10) * 10 + 9}")

    # Likelihood bucket
    lkh = play.get("likelihood_pct", 50)
    _record("likelihood_bucket", f"{(int(lkh) // 10) * 10}s")

    # DTE bucket
    dte = play.get("dte", 30)
    if dte <= 7:    _record("dte_bucket", "0-7")
    elif dte <= 14: _record("dte_bucket", "8-14")
    elif dte <= 21: _record("dte_bucket", "15-21")
    elif dte <= 35: _record("dte_bucket", "22-35")
    elif dte <= 45: _record("dte_bucket", "36-45")
    else:           _record("dte_bucket", "45+")

    # Delta bucket
    delta = abs(play.get("delta", 0.4))
    if delta < 0.2:   _record("delta_bucket", "<0.20")
    elif delta < 0.3: _record("delta_bucket", "0.20-0.29")
    elif delta < 0.4: _record("delta_bucket", "0.30-0.39")
    elif delta < 0.5: _record("delta_bucket", "0.40-0.49")
    elif delta < 0.6: _record("delta_bucket", "0.50-0.59")
    else:             _record("delta_bucket", "0.60+")

    # GEX confluence present
    gex = play.get("gex_confluence", {})
    if isinstance(gex, dict) and gex.get("near_pin"):
        _record("gex_near_pin", "yes")
    if isinstance(gex, dict) and gex.get("near_wall"):
        _record("gex_near_wall", "yes")

    # Flow confluence
    flow = play.get("flow_confluence", {})
    if isinstance(flow, dict) and flow.get("confirmed"):
        _record("flow_confirmed", "yes")

    # Regime
    regime = play.get("regime", "unknown")
    _record("regime", regime)

    # Option type
    _record("type", play.get("type", "CALL"))


def _update_combo_stats(data: Dict, play: Dict, won: bool):
    cs = data.setdefault("combo_stats", {})
    signals = play.get("tech_signals", [])
    if not signals:
        return
    key = "+".join(sorted(str(s) for s in signals[:3]))
    if key not in cs:
        cs[key] = {"wins": 0, "total": 0, "signals": signals[:3]}
    cs[key]["total"] += 1
    if won:
        cs[key]["wins"] += 1


# ─────────────────────────────────────────────────────────────────
# Pattern similarity score — boost plays similar to past winners
# ─────────────────────────────────────────────────────────────────

def get_pattern_boost(play: Dict) -> Tuple[float, str]:
    """
    Compare a new play against historical winners.
    Returns (boost_score 0-15, reason_string).
    Used by scanner to adjust final score.
    """
    data    = _load()
    closed  = data.get("closed_plays", [])
    winners = [p for p in closed if p.get("result") in ("WIN", "PARTIAL")]

    if len(winners) < 3:
        return 0.0, ""

    score_total = 0.0
    matches     = []

    for w in winners[-50:]:   # look at last 50 winners
        sim = _similarity(play, w)
        if sim >= 0.65:
            score_total += sim
            matches.append(w)

    if not matches:
        return 0.0, ""

    avg_sim    = score_total / len(matches)
    boost      = round(min(15.0, avg_sim * 20), 1)
    avg_return = sum(m.get("return_pct", 0) for m in matches) / len(matches)

    reason = (
        f"Similar to {len(matches)} past winner{'s' if len(matches)>1 else ''} "
        f"(avg return: {avg_return:+.0f}%)"
    )
    return boost, reason


def _similarity(play_a: Dict, play_b: Dict) -> float:
    """Cosine-like similarity across key factors (0-1)."""
    def norm(v, lo, hi):
        return max(0.0, min(1.0, (v - lo) / (hi - lo))) if hi > lo else 0.5

    factors = [
        (norm(play_a.get("score", 0), 0, 100),
         norm(play_b.get("score", 0), 0, 100)),
        (norm(play_a.get("likelihood_pct", 50), 0, 100),
         norm(play_b.get("likelihood_pct", 50), 0, 100)),
        (norm(abs(play_a.get("tech_bias", 0)), 0, 100),
         norm(abs(play_b.get("tech_bias", 0)), 0, 100)),
        (norm(play_a.get("rr", 2), 0, 5),
         norm(play_b.get("rr", 2), 0, 5)),
        (norm(abs(play_a.get("delta", 0.4)), 0, 1),
         norm(abs(play_b.get("delta", 0.4)), 0, 1)),
        (norm(play_a.get("dte", 30), 0, 60),
         norm(play_b.get("dte", 30), 0, 60)),
    ]
    # Type match bonus
    type_match = 1.0 if play_a.get("type") == play_b.get("type") else 0.5

    dot  = sum(a * b for a, b in factors) * type_match
    mag  = math.sqrt(sum(a**2 for a, _ in factors)) * math.sqrt(sum(b**2 for _, b in factors))
    return round(dot / mag, 3) if mag > 0 else 0.0


# ─────────────────────────────────────────────────────────────────
# Learned weights for scanner
# ─────────────────────────────────────────────────────────────────

def get_weights(regime: str = None) -> Dict[str, float]:
    """Get current learned weights, optionally regime-specific."""
    data = _load()
    if regime and regime in data.get("regime_weights", {}):
        rw = data["regime_weights"][regime]
        if sum(rw.values()) > 0:
            return rw
    return data.get("weights", DEFAULT_WEIGHTS.copy())


# ─────────────────────────────────────────────────────────────────
# Insights generator
# ─────────────────────────────────────────────────────────────────

def _generate_insights(data: Dict) -> List[Dict]:
    insights = []
    closed   = data.get("closed_plays", [])
    if not closed:
        return insights

    wins   = [p for p in closed if p.get("result") == "WIN"]
    losses = [p for p in closed if p.get("result") == "LOSS"]

    # Best score range
    if wins and losses:
        avg_w = sum(p["score"] for p in wins) / len(wins)
        avg_l = sum(p["score"] for p in losses) / len(losses)
        insights.append({
            "type": "SCORE_THRESHOLD",
            "message": f"Winners avg score {avg_w:.0f} vs losers {avg_l:.0f}. Min score should be ≥{max(40, int(avg_l)+5)}.",
            "data": {"avg_win_score": avg_w, "avg_loss_score": avg_l},
        })

    # Best DTE
    win_dtes = [p["dte"] for p in wins if "dte" in p]
    if len(win_dtes) >= 3:
        avg_dte = sum(win_dtes) / len(win_dtes)
        insights.append({
            "type": "OPTIMAL_DTE",
            "message": f"Your winning trades avg {avg_dte:.0f} DTE. Focus on {int(avg_dte-4)}–{int(avg_dte+4)} DTE.",
            "data": {"avg_winning_dte": avg_dte},
        })

    # GEX confluence impact
    gex_wins   = [p for p in wins   if isinstance(p.get("gex_confluence"), dict) and p["gex_confluence"].get("near_pin")]
    gex_losses = [p for p in losses if isinstance(p.get("gex_confluence"), dict) and p["gex_confluence"].get("near_pin")]
    if len(gex_wins) + len(gex_losses) >= 4:
        total_gex = len(gex_wins) + len(gex_losses)
        gex_wr = len(gex_wins) / total_gex * 100
        insights.append({
            "type": "GEX_CONFLUENCE",
            "message": f"GEX pin confluence plays: {gex_wr:.0f}% win rate ({len(gex_wins)}W/{len(gex_losses)}L).",
            "data": {"gex_win_rate": gex_wr},
        })

    # Flow confluence impact
    flow_wins   = [p for p in wins   if isinstance(p.get("flow_confluence"), dict) and p["flow_confluence"].get("confirmed")]
    flow_losses = [p for p in losses if isinstance(p.get("flow_confluence"), dict) and p["flow_confluence"].get("confirmed")]
    if len(flow_wins) + len(flow_losses) >= 4:
        total_f = len(flow_wins) + len(flow_losses)
        flow_wr = len(flow_wins) / total_f * 100
        insights.append({
            "type": "FLOW_CONFLUENCE",
            "message": f"Flow-confirmed plays: {flow_wr:.0f}% win rate ({len(flow_wins)}W/{len(flow_losses)}L).",
            "data": {"flow_win_rate": flow_wr},
        })

    # Weight shifts
    current  = data["weights"]
    shifted  = [(k, v - DEFAULT_WEIGHTS[k]) for k, v in current.items() if abs(v - DEFAULT_WEIGHTS[k]) > 0.025]
    shifted.sort(key=lambda x: abs(x[1]), reverse=True)
    if shifted:
        k, d = shifted[0]
        direction = "more" if d > 0 else "less"
        insights.append({
            "type": "WEIGHT_SHIFT",
            "message": f"AI now weights '{k.replace('_',' ')}' {direction} based on your trade history.",
            "data": {"factor": k, "shift": round(d, 4)},
        })

    return insights[-10:]


# ─────────────────────────────────────────────────────────────────
# Read queries
# ─────────────────────────────────────────────────────────────────

def get_state() -> Dict:
    data = _load()
    fs   = data.get("factor_stats", {})

    # Top performing factor buckets
    top_factors = sorted(
        [{"bucket": k, "win_rate": round(v["wins"]/v["total"]*100,1), "total": v["total"]}
         for k, v in fs.items() if v["total"] >= 3],
        key=lambda x: x["win_rate"], reverse=True
    )[:10]

    # Top combos
    cs = data.get("combo_stats", {})
    top_combos = sorted(
        [{"signals": v["signals"], "win_rate": round(v["wins"]/v["total"]*100,1), "total": v["total"]}
         for k, v in cs.items() if v["total"] >= 2],
        key=lambda x: x["win_rate"], reverse=True
    )[:5]

    return {
        "weights":         data["weights"],
        "default_weights": DEFAULT_WEIGHTS,
        "weight_shifts":   {k: round(data["weights"].get(k,0) - DEFAULT_WEIGHTS[k], 4) for k in DEFAULT_WEIGHTS},
        "stats":           data["stats"],
        "open_plays":      [p for p in data["open_plays"] if p.get("tracked")],
        "surfaced_today":  _surfaced_today(data),
        "recent_closed":   data["closed_plays"][-20:],
        "insights":        data["insights"],
        "top_factor_buckets": top_factors,
        "top_combos":      top_combos,
        "total_history":   len(data["open_plays"]) + len(data["closed_plays"]),
    }


def get_open_plays() -> List[Dict]:
    return [p for p in _load()["open_plays"] if p.get("tracked")]


def get_surfaced_plays(limit: int = 50) -> List[Dict]:
    data = _load()
    return sorted(data["open_plays"], key=lambda p: p["surfaced_date"], reverse=True)[:limit]


def _surfaced_today(data: Dict) -> int:
    today = _now()[:10]
    return sum(1 for p in data["open_plays"] if p.get("surfaced_date","")[:10] == today)


def reset_weights() -> Dict:
    data = _load()
    data["weights"] = DEFAULT_WEIGHTS.copy()
    for regime in data.get("regime_weights", {}):
        data["regime_weights"][regime] = DEFAULT_WEIGHTS.copy()
    _save(data)
    return {"success": True, "weights": DEFAULT_WEIGHTS}


def _make_id(play: Dict) -> str:
    return (
        f"{play.get('ticker','')}_{play.get('type','')}_{play.get('strike',0)}"
        f"_{play.get('expiry','')}_{_now()[:10]}"
    ).replace(" ", "_")
