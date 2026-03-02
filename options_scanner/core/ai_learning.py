"""
ai_learning.py
--------------
JuicyScanner AI Learning Engine

Tracks play outcomes (win/loss/partial) and uses them to:
  1. Adjust factor weights in the scoring model over time
  2. Learn which indicators are most predictive per market regime
  3. Surface "confidence patterns" — setups that historically perform well
  4. Penalize setups that have historically failed

Storage: JSON file (swap to SQLite/DB when you have many plays)
Learning method: Exponential moving average weight adjustment
              + Bayesian success rate per indicator combination
"""

import json
import os
import math
from datetime import datetime
from typing import Dict, List, Optional, Any

LEARNING_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "ai_learning.json")

# ─── Default factor weights (must sum to 1.0) ───────────────────
DEFAULT_WEIGHTS = {
    "rr_ratio":      0.20,
    "target_move":   0.15,
    "iv_rank":       0.15,
    "delta":         0.10,
    "liquidity":     0.10,
    "dte":           0.10,
    "likelihood":    0.15,
    "tech_bias":     0.05,
}

# ─── Learning rate (how fast weights shift per outcome) ──────────
LEARNING_RATE = 0.05


# ─────────────────────────────────────────────────────────────────
# Storage helpers
# ─────────────────────────────────────────────────────────────────

def _load() -> Dict:
    if os.path.exists(LEARNING_FILE):
        with open(LEARNING_FILE, "r") as f:
            return json.load(f)
    return _default_state()


def _save(data: Dict):
    os.makedirs(os.path.dirname(LEARNING_FILE), exist_ok=True)
    with open(LEARNING_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _now() -> str:
    return datetime.now().isoformat()


def _default_state() -> Dict:
    return {
        "version":       "1.0",
        "created_at":    _now(),
        "weights":       DEFAULT_WEIGHTS.copy(),
        "plays":         [],          # logged plays
        "outcomes":      [],          # closed plays with results
        "stats": {
            "total_logged":   0,
            "total_closed":   0,
            "wins":           0,
            "losses":         0,
            "partials":       0,
            "win_rate":       0.0,
            "avg_return_pct": 0.0,
        },
        "indicator_stats": {
            # Tracks win rate per indicator signal combination
            # e.g. "rsi_bullish+macd_bullish_cross": {"wins": 5, "total": 7}
        },
        "regime_weights": {
            # Weights per market regime (future feature)
            "trending_bull":  DEFAULT_WEIGHTS.copy(),
            "trending_bear":  DEFAULT_WEIGHTS.copy(),
            "range_bound":    DEFAULT_WEIGHTS.copy(),
            "high_vol":       DEFAULT_WEIGHTS.copy(),
        },
        "insights": [],    # auto-generated learning insights
    }


# ─────────────────────────────────────────────────────────────────
# Log a new play (when you enter a trade)
# ─────────────────────────────────────────────────────────────────

def log_play(
    ticker: str,
    option_type: str,        # CALL / PUT
    strike: float,
    expiry: str,
    premium: float,
    score: int,
    rr: float,
    likelihood_pct: float,
    tech_bias: float,
    iv_rank: int,
    delta: float,
    target_pct: float,
    dte: int,
    tech_signals: List[str] = None,
    notes: str = "",
) -> Dict:
    """Record a play when you enter it. Returns the play_id to use when closing."""
    data    = _load()
    play_id = f"{ticker}_{option_type}_{strike}_{expiry}_{_now()[:10]}".replace(" ", "_")

    play = {
        "id":             play_id,
        "ticker":         ticker,
        "option_type":    option_type,
        "strike":         strike,
        "expiry":         expiry,
        "premium":        premium,
        "score":          score,
        "rr":             rr,
        "likelihood_pct": likelihood_pct,
        "tech_bias":      tech_bias,
        "iv_rank":        iv_rank,
        "delta":          delta,
        "target_pct":     target_pct,
        "dte":            dte,
        "tech_signals":   tech_signals or [],
        "notes":          notes,
        "logged_at":      _now(),
        "status":         "OPEN",
        "outcome":        None,
    }

    data["plays"].append(play)
    data["stats"]["total_logged"] += 1
    _save(data)

    return {"success": True, "play_id": play_id, "message": f"Play logged: {ticker} {option_type} ${strike} {expiry}"}


# ─────────────────────────────────────────────────────────────────
# Close a play with outcome (the learning trigger)
# ─────────────────────────────────────────────────────────────────

def close_play(
    play_id: str,
    exit_premium: float,
    result: str,           # "WIN", "LOSS", "PARTIAL", "EXPIRED"
    notes: str = "",
) -> Dict:
    """
    Close a play with its outcome. This triggers weight adjustment.
    result:
      WIN     = hit target (100%+ gain on premium)
      PARTIAL = closed early for profit (1-99% gain)
      LOSS    = stopped out (-50% or worse)
      EXPIRED = expired worthless
    """
    data = _load()

    # Find the play
    play = next((p for p in data["plays"] if p["id"] == play_id), None)
    if not play:
        return {"success": False, "error": f"Play {play_id} not found."}
    if play["status"] != "OPEN":
        return {"success": False, "error": f"Play {play_id} is already closed."}

    # Calculate return
    entry     = play["premium"]
    ret_pct   = round((exit_premium - entry) / entry * 100, 2)
    won       = result in ("WIN", "PARTIAL")

    # Update play record
    play["status"]       = "CLOSED"
    play["exit_premium"] = exit_premium
    play["return_pct"]   = ret_pct
    play["result"]       = result
    play["closed_at"]    = _now()
    play["close_notes"]  = notes
    play["outcome"]      = {
        "result":      result,
        "return_pct":  ret_pct,
        "won":         won,
    }

    # Add to outcomes list
    data["outcomes"].append(play)
    data["plays"] = [p for p in data["plays"] if p["id"] != play_id]

    # Update aggregate stats
    s = data["stats"]
    s["total_closed"] += 1
    if result == "WIN":          s["wins"]     += 1
    elif result == "PARTIAL":    s["partials"] += 1
    else:                        s["losses"]   += 1

    total = s["total_closed"]
    s["win_rate"]       = round((s["wins"] + s["partials"] * 0.5) / total * 100, 1)
    all_returns         = [o["return_pct"] for o in data["outcomes"] if "return_pct" in o]
    s["avg_return_pct"] = round(sum(all_returns) / len(all_returns), 1) if all_returns else 0.0

    # ── Weight adjustment ────────────────────────────────────────
    data["weights"] = _adjust_weights(data["weights"], play, won, ret_pct)

    # ── Indicator stats update ───────────────────────────────────
    _update_indicator_stats(data, play, won)

    # ── Generate insights if enough data ─────────────────────────
    if total >= 5 and total % 5 == 0:
        data["insights"] = _generate_insights(data)

    _save(data)
    return {
        "success":    True,
        "play_id":    play_id,
        "result":     result,
        "return_pct": ret_pct,
        "new_weights": data["weights"],
        "message":    f"Play closed ({result}, {ret_pct:+.1f}%). Weights updated.",
    }


# ─────────────────────────────────────────────────────────────────
# Weight adjustment engine
# ─────────────────────────────────────────────────────────────────

def _adjust_weights(
    weights: Dict[str, float],
    play: Dict,
    won: bool,
    return_pct: float,
) -> Dict[str, float]:
    """
    Adjust weights based on which factors were strong/weak in this play.
    
    Logic:
    - If the play WON and a factor was high for this play → increase that weight
    - If the play LOST and a factor was high → decrease that weight
    - Magnitude of adjustment scales with how extreme the win/loss was
    """
    # How strong was this result? Scale learning by outcome magnitude
    magnitude = min(2.0, abs(return_pct) / 50.0)  # 50% move = 1.0 magnitude
    lr        = LEARNING_RATE * magnitude
    sign      = 1.0 if won else -1.0

    # Factor scores from this play (normalized 0-1)
    factor_scores = {
        "rr_ratio":    min(1.0, play.get("rr", 2) / 5.0),
        "target_move": max(0, 1.0 - play.get("target_pct", 30) / 50.0),
        "iv_rank":     1.0 - abs(play.get("iv_rank", 50) - 40) / 60.0,
        "delta":       1.0 - abs(play.get("delta", 0.4) - 0.4) / 0.5,
        "liquidity":   0.7,  # assume decent (no OI stored in brief play record)
        "dte":         1.0 - abs(play.get("dte", 28) - 28) / 45.0,
        "likelihood":  play.get("likelihood_pct", 50) / 100.0,
        "tech_bias":   (abs(play.get("tech_bias", 0)) / 100.0),
    }

    new_weights = {}
    for factor, w in weights.items():
        fs     = factor_scores.get(factor, 0.5)
        # Factors that were strong in this play get reinforced/penalized
        delta  = lr * sign * (fs - 0.5)  # neutral factor = no change
        new_w  = max(0.02, min(0.40, w + delta))
        new_weights[factor] = round(new_w, 4)

    # Renormalize so weights sum to 1.0
    total = sum(new_weights.values())
    return {k: round(v / total, 4) for k, v in new_weights.items()}


# ─────────────────────────────────────────────────────────────────
# Indicator stats
# ─────────────────────────────────────────────────────────────────

def _update_indicator_stats(data: Dict, play: Dict, won: bool):
    """Track win rates per technical signal combination."""
    signals = play.get("tech_signals", [])
    if not signals:
        return

    # Create a key from the signal combination
    key = "+".join(sorted(str(s) for s in signals[:3]))  # top 3 signals
    if key not in data["indicator_stats"]:
        data["indicator_stats"][key] = {"wins": 0, "total": 0, "signals": signals[:3]}
    data["indicator_stats"][key]["total"] += 1
    if won:
        data["indicator_stats"][key]["wins"] += 1


# ─────────────────────────────────────────────────────────────────
# Insights generator
# ─────────────────────────────────────────────────────────────────

def _generate_insights(data: Dict) -> List[Dict]:
    """Auto-generate human-readable insights from learning data."""
    insights = []
    outcomes = data["outcomes"]
    if not outcomes:
        return insights

    wins   = [o for o in outcomes if o.get("result") == "WIN"]
    losses = [o for o in outcomes if o.get("result") == "LOSS"]

    # Insight 1: Best score range
    if len(outcomes) >= 5:
        win_scores  = [o["score"] for o in wins]   if wins   else []
        loss_scores = [o["score"] for o in losses] if losses else []
        if win_scores and loss_scores:
            avg_win_score  = sum(win_scores)  / len(win_scores)
            avg_loss_score = sum(loss_scores) / len(loss_scores)
            insights.append({
                "type":    "SCORE_THRESHOLD",
                "message": f"Winning plays averaged a score of {avg_win_score:.0f} vs {avg_loss_score:.0f} for losers. Consider raising your minimum score.",
                "data":    {"avg_win_score": avg_win_score, "avg_loss_score": avg_loss_score},
            })

    # Insight 2: Best DTE range
    win_dtes = [o["dte"] for o in wins if "dte" in o]
    if len(win_dtes) >= 3:
        avg_dte = sum(win_dtes) / len(win_dtes)
        insights.append({
            "type":    "OPTIMAL_DTE",
            "message": f"Your winning trades had an average DTE of {avg_dte:.0f} days. Focus on {int(avg_dte-5)}–{int(avg_dte+5)} DTE.",
            "data":    {"avg_winning_dte": avg_dte},
        })

    # Insight 3: IV rank sweet spot
    win_ivrs = [o["iv_rank"] for o in wins if "iv_rank" in o]
    if len(win_ivrs) >= 3:
        avg_ivr = sum(win_ivrs) / len(win_ivrs)
        insights.append({
            "type":    "OPTIMAL_IV_RANK",
            "message": f"Your winners had avg IV rank of {avg_ivr:.0f}. Setups with IV rank near {avg_ivr:.0f} are your sweet spot.",
            "data":    {"avg_winning_iv_rank": avg_ivr},
        })

    # Insight 4: Dominant weight shifts
    current  = data["weights"]
    defaults = DEFAULT_WEIGHTS
    shifted  = []
    for k, v in current.items():
        diff = v - defaults[k]
        if abs(diff) > 0.03:
            shifted.append((k, diff))
    shifted.sort(key=lambda x: abs(x[1]), reverse=True)
    if shifted:
        top = shifted[0]
        direction = "more" if top[1] > 0 else "less"
        insights.append({
            "type":    "WEIGHT_SHIFT",
            "message": f"The AI has learned to weight '{top[0].replace('_',' ')}' {direction} heavily based on your trade history.",
            "data":    {"factor": top[0], "shift": top[1]},
        })

    return insights[-10:]  # keep last 10 insights


# ─────────────────────────────────────────────────────────────────
# Read-only queries
# ─────────────────────────────────────────────────────────────────

def get_state() -> Dict:
    """Full learning state — weights, stats, insights, open plays."""
    data = _load()
    return {
        "weights":         data["weights"],
        "default_weights": DEFAULT_WEIGHTS,
        "weight_shifts":   {k: round(data["weights"][k] - DEFAULT_WEIGHTS[k], 4) for k in DEFAULT_WEIGHTS},
        "stats":           data["stats"],
        "open_plays":      data["plays"],
        "recent_outcomes": data["outcomes"][-20:],
        "insights":        data["insights"],
        "indicator_stats": _top_indicator_stats(data),
        "total_plays":     len(data["plays"]) + len(data["outcomes"]),
    }


def get_adjusted_weights() -> Dict[str, float]:
    """Returns the current learned weights for use by the scanner."""
    return _load()["weights"]


def _top_indicator_stats(data: Dict) -> List[Dict]:
    """Return top 5 most successful indicator combos."""
    stats = []
    for key, val in data["indicator_stats"].items():
        if val["total"] >= 2:
            wr = val["wins"] / val["total"] * 100
            stats.append({
                "signals":   val.get("signals", []),
                "win_rate":  round(wr, 1),
                "total":     val["total"],
                "wins":      val["wins"],
            })
    stats.sort(key=lambda x: x["win_rate"], reverse=True)
    return stats[:5]


def get_open_plays() -> List[Dict]:
    return _load()["plays"]


def reset_weights() -> Dict:
    """Reset weights back to defaults (nuclear option)."""
    data = _load()
    data["weights"] = DEFAULT_WEIGHTS.copy()
    _save(data)
    return {"success": True, "message": "Weights reset to defaults.", "weights": DEFAULT_WEIGHTS}
