"""
brain.py
--------
JuicyScanner Unified AI Brain

Single source of truth for:
  - Auto-logging every surfaced play from scanner
  - Tracking entered trades + outcomes
  - Learning: weight rebalancing from outcomes
  - Insights: statistical patterns from history
  - Pattern boost: similarity score vs past winners

Replaces fragmented ai_learning.py + play_history.py
"""

import json
import os
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

_LOCAL = os.path.join(os.path.dirname(__file__), "..", "data", "brain.json")
BRAIN_FILE = "/app/brain.json" if os.path.exists("/app") else _LOCAL

# ── Default scoring weights (must sum to 1.0) ─────────────────────
DEFAULT_WEIGHTS = {
    "likelihood":  0.20,
    "tech_bias":   0.15,
    "gex_conf":    0.12,
    "flow_conf":   0.12,
    "rr_ratio":    0.12,
    "iv_rank":     0.10,
    "delta":       0.08,
    "dte":         0.06,
    "liquidity":   0.05,
}

LEARNING_RATE = 0.04
MIN_WEIGHT    = 0.02
MAX_WEIGHT    = 0.40

FACTOR_LABELS = {
    "likelihood": "Likelihood Score",
    "tech_bias":  "Technical Bias",
    "gex_conf":   "GEX Confluence",
    "flow_conf":  "Flow Confluence",
    "rr_ratio":   "Risk:Reward",
    "iv_rank":    "IV Rank",
    "delta":      "Delta",
    "dte":        "Days to Expiry",
    "liquidity":  "Liquidity",
}


# ─────────────────────────────────────────────────────────────────
# Storage
# ─────────────────────────────────────────────────────────────────

def _load() -> Dict:
    if os.path.exists(BRAIN_FILE):
        try:
            with open(BRAIN_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return _default_state()


def _save(data: Dict):
    try:
        os.makedirs(os.path.dirname(BRAIN_FILE), exist_ok=True)
        with open(BRAIN_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[Brain] save error: {e}")


def _now() -> str:
    return datetime.now().isoformat()


def _default_state() -> Dict:
    return {
        "version":    "3.0",
        "created_at": _now(),

        # Learned weights — updated after every closed play
        "weights": DEFAULT_WEIGHTS.copy(),
        "regime_weights": {
            "bull":     DEFAULT_WEIGHTS.copy(),
            "bear":     DEFAULT_WEIGHTS.copy(),
            "chop":     DEFAULT_WEIGHTS.copy(),
            "high_vol": DEFAULT_WEIGHTS.copy(),
        },

        # All plays ever surfaced by the scanner
        # status: SURFACED → TRACKING → CLOSED
        "plays": [],

        # Aggregate performance stats
        "stats": {
            "total_surfaced": 0,
            "total_tracking": 0,
            "total_closed":   0,
            "wins": 0, "losses": 0, "partials": 0, "expired": 0,
            "win_rate":       0.0,
            "avg_return_pct": 0.0,
            "avg_win_pct":    0.0,
            "avg_loss_pct":   0.0,
            "profit_factor":  0.0,
            "expectancy":     0.0,   # avg $ outcome per $1 risked
        },

        # Statistical breakdowns for insight generation
        "factor_stats": {},   # win rate per factor bucket
        "combo_stats":  {},   # win rate per indicator combo
        "regime_stats": {},   # win rate per regime

        # Auto-generated insights (updated every 5 closes)
        "insights": [],
    }


# ─────────────────────────────────────────────────────────────────
# Auto-log every surfaced play (called by scanner)
# ─────────────────────────────────────────────────────────────────

def auto_log_scan(plays: List[Dict], scan_context: Dict = None) -> int:
    """
    Called after every scan. Logs all surfaced plays.
    Skips duplicates (same ticker/strike/expiry surfaced today).
    Returns count of new plays logged.
    """
    if not plays:
        return 0

    data    = _load()
    saved   = 0
    context = scan_context or {}
    today   = _now()[:10]

    existing_ids = {p["id"] for p in data["plays"]}

    for play in plays:
        pid = _make_id(play)
        if pid in existing_ids:
            continue

        gex  = play.get("gex_confluence", {}) or {}
        flow = play.get("flow_confluence", {}) or {}

        record = {
            "id":            pid,
            "status":        "SURFACED",
            "surfaced_at":   _now(),
            "tracking_at":   None,
            "closed_at":     None,

            # Identity
            "ticker":        play.get("ticker", ""),
            "type":          play.get("type", "CALL"),
            "strike":        play.get("strike", 0),
            "expiry":        play.get("expiry", ""),
            "dte":           play.get("dte", 0),
            "premium":       play.get("premium", 0),
            "entry_premium": None,   # set when user starts tracking
            "exit_premium":  None,
            "underlying_px": play.get("underlyingPx", 0),

            # Scores at scan time
            "score":          play.get("score", 0),
            "likelihood_pct": play.get("likelihood", {}).get("likelihood_pct", 50),
            "rr":             play.get("rr", 0),
            "target_pct":     play.get("target", 0),
            "iv_rank":        play.get("ivRank", 50),
            "delta":          abs(play.get("delta", 0.3)),
            "tech_bias":      play.get("tech_bias", 0),

            # Confluence at scan time
            "gex_node":       gex.get("node", ""),
            "gex_score":      gex.get("score", 0),
            "gex_flip":       gex.get("flip_zone", None),
            "flow_confirmed": flow.get("confirmed", False),
            "flow_score":     flow.get("score", 0),
            "tech_signals":   [s.get("label", "") if isinstance(s, dict) else str(s)
                               for s in play.get("tech_signals", [])],

            # Context
            "regime":         context.get("regime", "unknown"),

            # Outcome
            "result":         None,   # WIN / PARTIAL / LOSS / EXPIRED
            "return_pct":     None,
            "notes":          "",
        }

        data["plays"].append(record)
        existing_ids.add(pid)
        data["stats"]["total_surfaced"] += 1
        saved += 1

    _save(data)
    return saved


# ─────────────────────────────────────────────────────────────────
# Start tracking a play (user decides to enter)
# ─────────────────────────────────────────────────────────────────

def start_tracking(play_id: str, entry_premium: float = None, notes: str = "") -> Dict:
    """Mark a surfaced play as being actively tracked (you entered the trade)."""
    data = _load()
    play = next((p for p in data["plays"] if p["id"] == play_id), None)
    if not play:
        return {"success": False, "error": f"Play {play_id} not found"}
    if play["status"] == "CLOSED":
        return {"success": False, "error": "Play already closed"}

    play["status"]        = "TRACKING"
    play["tracking_at"]   = _now()
    play["entry_premium"] = entry_premium or play["premium"]
    play["notes"]         = notes

    data["stats"]["total_tracking"] += 1
    _save(data)
    return {"success": True, "play_id": play_id}


# ─────────────────────────────────────────────────────────────────
# Close a play → triggers learning
# ─────────────────────────────────────────────────────────────────

def close_play(play_id: str, exit_premium: float, result: str, notes: str = "") -> Dict:
    """
    Close a tracked play with its outcome.
    This is the main learning trigger — weights adjust based on what worked.
    """
    data = _load()
    play = next((p for p in data["plays"] if p["id"] == play_id), None)
    if not play:
        return {"success": False, "error": f"Play {play_id} not found"}

    entry   = play.get("entry_premium") or play["premium"]
    ret_pct = round((exit_premium - entry) / entry * 100, 2) if entry > 0 else 0.0
    won     = result in ("WIN", "PARTIAL")

    play.update({
        "status":        "CLOSED",
        "closed_at":     _now(),
        "exit_premium":  exit_premium,
        "return_pct":    ret_pct,
        "result":        result,
        "notes":         notes,
    })

    # ── Update aggregate stats ────────────────────────────────────
    s = data["stats"]
    s["total_closed"] += 1
    if result == "WIN":       s["wins"]     += 1
    elif result == "PARTIAL": s["partials"] += 1
    elif result == "EXPIRED": s["expired"]  += 1
    else:                     s["losses"]   += 1

    total   = s["total_closed"]
    closed  = [p for p in data["plays"] if p.get("result") is not None]
    returns = [p["return_pct"] for p in closed if p.get("return_pct") is not None]

    s["win_rate"]       = round((s["wins"] + s["partials"] * 0.5) / total * 100, 1) if total else 0
    s["avg_return_pct"] = round(sum(returns) / len(returns), 1) if returns else 0

    wins_r   = [r for r in returns if r > 0]
    losses_r = [r for r in returns if r <= 0]
    s["avg_win_pct"]    = round(sum(wins_r)   / len(wins_r),   1) if wins_r   else 0
    s["avg_loss_pct"]   = round(sum(losses_r) / len(losses_r), 1) if losses_r else 0

    gross_wins   = sum(wins_r)           if wins_r   else 0
    gross_losses = abs(sum(losses_r))    if losses_r else 0
    s["profit_factor"]  = round(gross_wins / gross_losses, 2) if gross_losses else 0

    # Expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)
    wr = s["win_rate"] / 100
    s["expectancy"] = round(wr * s["avg_win_pct"] + (1 - wr) * s["avg_loss_pct"], 1)

    # ── Learning: adjust weights ──────────────────────────────────
    data["weights"] = _adjust_weights(data["weights"], play, won, ret_pct)
    regime = play.get("regime", "unknown")
    if regime in data["regime_weights"]:
        data["regime_weights"][regime] = _adjust_weights(
            data["regime_weights"][regime], play, won, ret_pct
        )

    # ── Update factor / combo stats ───────────────────────────────
    _update_factor_stats(data, play, won, ret_pct)
    _update_combo_stats(data, play, won)

    # ── Regenerate insights every 5 closes ────────────────────────
    if total >= 5 and total % 5 == 0:
        data["insights"] = _generate_insights(data)

    _save(data)
    return {
        "success":     True,
        "play_id":     play_id,
        "result":      result,
        "return_pct":  ret_pct,
        "new_weights": data["weights"],
    }


def update_entry(play_id: str, entry_premium: float = None, notes: str = None) -> Dict:
    """Update entry price or notes on a tracked play."""
    data = _load()
    play = next((p for p in data["plays"] if p["id"] == play_id), None)
    if not play:
        return {"success": False, "error": "Play not found"}
    if entry_premium is not None:
        play["entry_premium"] = round(entry_premium, 2)
    if notes is not None:
        play["notes"] = notes
    _save(data)
    return {"success": True, "play": play}


# ─────────────────────────────────────────────────────────────────
# Weight learning engine
# ─────────────────────────────────────────────────────────────────

def _adjust_weights(weights: Dict, play: Dict, won: bool, return_pct: float) -> Dict:
    """
    EMA weight update.
    Factors that were strong in this play get boosted on a win, penalised on a loss.
    Magnitude scales with size of return (bigger wins/losses = faster learning).
    """
    magnitude = min(2.0, abs(return_pct) / 50.0)
    lr        = LEARNING_RATE * magnitude
    sign      = 1.0 if won else -1.0

    factor_scores = {
        "likelihood": play.get("likelihood_pct", 50) / 100.0,
        "tech_bias":  min(1.0, abs(play.get("tech_bias", 0)) / 100.0),
        "gex_conf":   min(1.0, play.get("gex_score", 0) / 100.0),
        "flow_conf":  min(1.0, play.get("flow_score", 0) / 100.0),
        "rr_ratio":   min(1.0, play.get("rr", 2) / 5.0),
        "iv_rank":    1.0 - abs((play.get("iv_rank", 40) - 35)) / 65.0,
        "delta":      1.0 - abs((play.get("delta", 0.4) - 0.4)) / 0.5,
        "dte":        1.0 - abs((play.get("dte", 28) - 28)) / 45.0,
        "liquidity":  0.6,
    }

    new_w = {}
    for k, w in weights.items():
        fs    = factor_scores.get(k, 0.5)
        adj   = lr * sign * (fs - 0.5)
        new_w[k] = round(max(MIN_WEIGHT, min(MAX_WEIGHT, w + adj)), 4)

    total = sum(new_w.values())
    return {k: round(v / total, 4) for k, v in new_w.items()}


# ─────────────────────────────────────────────────────────────────
# Factor + combo stats
# ─────────────────────────────────────────────────────────────────

def _update_factor_stats(data: Dict, play: Dict, won: bool, ret_pct: float):
    fs = data.setdefault("factor_stats", {})

    def rec(key, val):
        b = f"{key}:{val}"
        if b not in fs:
            fs[b] = {"wins": 0, "total": 0, "total_return": 0.0}
        fs[b]["total"]        += 1
        fs[b]["total_return"] += ret_pct
        if won:
            fs[b]["wins"] += 1

    score = play.get("score", 0)
    rec("score_bucket", f"{(score//10)*10}-{(score//10)*10+9}")

    lkh = play.get("likelihood_pct", 50)
    rec("likelihood_bucket", f"{(int(lkh)//10)*10}s")

    dte = play.get("dte", 30)
    if dte <= 7:    rec("dte", "0-7d")
    elif dte <= 14: rec("dte", "8-14d")
    elif dte <= 21: rec("dte", "15-21d")
    elif dte <= 35: rec("dte", "22-35d")
    elif dte <= 45: rec("dte", "36-45d")
    else:           rec("dte", "45d+")

    d = play.get("delta", 0.4)
    if d < 0.2:     rec("delta", "<0.20")
    elif d < 0.3:   rec("delta", "0.20-0.29")
    elif d < 0.4:   rec("delta", "0.30-0.39")
    elif d < 0.5:   rec("delta", "0.40-0.49")
    else:           rec("delta", "0.50+")

    if play.get("gex_node"):
        rec("gex_node", play["gex_node"])
    if play.get("flow_confirmed"):
        rec("flow_confirmed", "yes")

    rec("regime", play.get("regime", "unknown"))
    rec("type",   play.get("type", "CALL"))


def _update_combo_stats(data: Dict, play: Dict, won: bool):
    cs  = data.setdefault("combo_stats", {})
    sig = play.get("tech_signals", [])
    if not sig:
        return
    key = "+".join(sorted(str(s) for s in sig[:4]))
    if key not in cs:
        cs[key] = {"wins": 0, "total": 0, "signals": sig[:4]}
    cs[key]["total"] += 1
    if won:
        cs[key]["wins"] += 1


# ─────────────────────────────────────────────────────────────────
# Pattern boost — used by scanner to boost plays similar to winners
# ─────────────────────────────────────────────────────────────────

def get_pattern_boost(play: Dict) -> Tuple[float, str]:
    """Returns (boost 0-15, reason_string) based on similarity to past winners."""
    data    = _load()
    winners = [p for p in data["plays"] if p.get("result") in ("WIN", "PARTIAL")]

    if len(winners) < 3:
        return 0.0, ""

    score_total = 0.0
    matches     = []

    for w in winners[-50:]:
        sim = _similarity(play, w)
        if sim >= 0.65:
            score_total += sim
            matches.append(w)

    if not matches:
        return 0.0, ""

    avg_sim    = score_total / len(matches)
    boost      = round(min(15.0, avg_sim * 20), 1)
    avg_return = sum(m.get("return_pct", 0) for m in matches) / len(matches)
    reason     = f"Similar to {len(matches)} past winner{'s' if len(matches)>1 else ''} (avg {avg_return:+.0f}%)"
    return boost, reason


def _similarity(a: Dict, b: Dict) -> float:
    def norm(v, lo, hi):
        return max(0.0, min(1.0, (v - lo) / (hi - lo))) if hi > lo else 0.5

    pairs = [
        (norm(a.get("score",0), 0, 100),          norm(b.get("score",0), 0, 100)),
        (norm(a.get("likelihood_pct",50), 0, 100), norm(b.get("likelihood_pct",50), 0, 100)),
        (norm(abs(a.get("tech_bias",0)), 0, 100),  norm(abs(b.get("tech_bias",0)), 0, 100)),
        (norm(a.get("rr",2), 0, 5),                norm(b.get("rr",2), 0, 5)),
        (norm(a.get("delta",0.4), 0, 1),           norm(b.get("delta",0.4), 0, 1)),
        (norm(a.get("dte",30), 0, 60),             norm(b.get("dte",30), 0, 60)),
    ]
    type_match = 1.0 if a.get("type") == b.get("type") else 0.5
    dot = sum(x*y for x,y in pairs) * type_match
    mag = math.sqrt(sum(x**2 for x,_ in pairs)) * math.sqrt(sum(y**2 for _,y in pairs))
    return round(dot / mag, 3) if mag > 0 else 0.0


# ─────────────────────────────────────────────────────────────────
# Learned weights (used by scanner to score plays)
# ─────────────────────────────────────────────────────────────────

def get_weights(regime: str = None) -> Dict[str, float]:
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
    closed   = [p for p in data["plays"] if p.get("result")]
    if not closed:
        return []

    wins   = [p for p in closed if p.get("result") == "WIN"]
    losses = [p for p in closed if p.get("result") == "LOSS"]
    all_r  = [p["return_pct"] for p in closed if p.get("return_pct") is not None]

    # Overall win rate
    wr = data["stats"].get("win_rate", 0)
    if wr > 0:
        insights.append({
            "type": "WIN_RATE",
            "icon": "📊",
            "message": f"Overall win rate {wr:.0f}% across {len(closed)} closed plays. "
                       f"Avg return: {sum(all_r)/len(all_r):+.0f}%." if all_r else "",
        })

    # Score threshold
    if wins and losses:
        avg_w = sum(p["score"] for p in wins) / len(wins)
        avg_l = sum(p["score"] for p in losses) / len(losses)
        threshold = max(40, int(avg_l) + 5)
        insights.append({
            "type": "SCORE_THRESHOLD",
            "icon": "🎯",
            "message": f"Winners avg score {avg_w:.0f}, losers {avg_l:.0f}. "
                       f"Consider minimum score threshold: {threshold}.",
        })

    # Best DTE range
    win_dtes = [p["dte"] for p in wins if "dte" in p]
    if len(win_dtes) >= 3:
        avg_dte = sum(win_dtes) / len(win_dtes)
        insights.append({
            "type": "OPTIMAL_DTE",
            "icon": "📅",
            "message": f"Winning trades avg {avg_dte:.0f} DTE. "
                       f"Sweet spot: {max(1,int(avg_dte)-5)}–{int(avg_dte)+5} DTE.",
        })

    # GEX node performance
    fs = data.get("factor_stats", {})
    gex_buckets = {k: v for k, v in fs.items() if k.startswith("gex_node:") and v["total"] >= 3}
    if gex_buckets:
        best = max(gex_buckets.items(), key=lambda x: x[1]["wins"] / x[1]["total"])
        node = best[0].split(":")[1]
        wr_n = best[1]["wins"] / best[1]["total"] * 100
        insights.append({
            "type": "GEX_NODE",
            "icon": "⚡",
            "message": f"Best GEX node: {node} — {wr_n:.0f}% win rate "
                       f"({best[1]['wins']}W / {best[1]['total'] - best[1]['wins']}L).",
        })

    # Flow confirmation
    flow_b = fs.get("flow_confirmed:yes", {})
    if flow_b.get("total", 0) >= 4:
        flow_wr = flow_b["wins"] / flow_b["total"] * 100
        insights.append({
            "type": "FLOW_CONF",
            "icon": "🌊",
            "message": f"Flow-confirmed plays: {flow_wr:.0f}% win rate "
                       f"({flow_b['wins']}W / {flow_b['total']-flow_b['wins']}L). "
                       f"{'Strong edge — prioritise these.' if flow_wr > 60 else 'Not yet a reliable filter.'}",
        })

    # Top shifted weight
    current = data["weights"]
    shifted = [(k, v - DEFAULT_WEIGHTS[k]) for k, v in current.items()]
    shifted.sort(key=lambda x: abs(x[1]), reverse=True)
    if shifted and abs(shifted[0][1]) > 0.02:
        k, d = shifted[0]
        direction = "↑ boosted" if d > 0 else "↓ reduced"
        insights.append({
            "type": "WEIGHT_SHIFT",
            "icon": "🧠",
            "message": f"AI {direction} weight for '{FACTOR_LABELS.get(k,k)}' "
                       f"({d:+.1%}) based on your trade history.",
        })

    # Profit factor
    pf = data["stats"].get("profit_factor", 0)
    if pf > 0:
        emoji = "🚀" if pf >= 2 else "✅" if pf >= 1 else "⚠️"
        insights.append({
            "type": "PROFIT_FACTOR",
            "icon": emoji,
            "message": f"Profit factor {pf:.2f}x "
                       f"({'excellent' if pf>=2 else 'positive' if pf>=1 else 'needs work'}).",
        })

    # Best regime
    reg_b = {k: v for k, v in fs.items() if k.startswith("regime:") and v["total"] >= 3}
    if len(reg_b) >= 2:
        best_r = max(reg_b.items(), key=lambda x: x[1]["wins"] / x[1]["total"])
        reg    = best_r[0].split(":")[1]
        wr_r   = best_r[1]["wins"] / best_r[1]["total"] * 100
        insights.append({
            "type": "BEST_REGIME",
            "icon": "📈",
            "message": f"Best performance in {reg} regime: {wr_r:.0f}% win rate.",
        })

    # Best combo
    cs = data.get("combo_stats", {})
    good_combos = [(k, v) for k, v in cs.items() if v["total"] >= 3 and v["wins"]/v["total"] >= 0.7]
    if good_combos:
        best_c = max(good_combos, key=lambda x: x[1]["wins"] / x[1]["total"])
        wr_c   = best_c[1]["wins"] / best_c[1]["total"] * 100
        sigs   = " + ".join(str(s) for s in best_c[1]["signals"][:2])
        insights.append({
            "type": "TOP_COMBO",
            "icon": "💡",
            "message": f"Signal combo '{sigs}' wins {wr_c:.0f}% "
                       f"({best_c[1]['wins']}/{best_c[1]['total']} plays).",
        })

    return insights[-12:]


# ─────────────────────────────────────────────────────────────────
# State query (for API + UI)
# ─────────────────────────────────────────────────────────────────

def get_state() -> Dict:
    data = _load()

    plays = data["plays"]
    today = _now()[:10]

    surfaced_today = [p for p in plays if p.get("surfaced_at","")[:10] == today]
    tracking       = [p for p in plays if p["status"] == "TRACKING"]
    closed_recent  = sorted(
        [p for p in plays if p["status"] == "CLOSED"],
        key=lambda p: p.get("closed_at",""), reverse=True
    )[:30]
    surfaced_recent = sorted(
        [p for p in plays if p["status"] == "SURFACED"],
        key=lambda p: p.get("surfaced_at",""), reverse=True
    )[:50]

    # Factor buckets with >= 3 plays
    fs = data.get("factor_stats", {})
    top_factors = sorted(
        [{"bucket": k, "win_rate": round(v["wins"]/v["total"]*100,1),
          "total": v["total"], "avg_return": round(v["total_return"]/v["total"],1)}
         for k, v in fs.items() if v["total"] >= 3],
        key=lambda x: x["win_rate"], reverse=True
    )[:12]

    cs = data.get("combo_stats", {})
    top_combos = sorted(
        [{"signals": v["signals"], "win_rate": round(v["wins"]/v["total"]*100,1),
          "total": v["total"]}
         for k, v in cs.items() if v["total"] >= 2],
        key=lambda x: x["win_rate"], reverse=True
    )[:8]

    w  = data["weights"]
    dw = DEFAULT_WEIGHTS
    return {
        "weights":          w,
        "default_weights":  dw,
        "weight_shifts":    {k: round(w.get(k,0) - dw[k], 4) for k in dw},
        "factor_labels":    FACTOR_LABELS,
        "stats":            data["stats"],
        "insights":         data.get("insights", []),
        "tracking_plays":   tracking,
        "surfaced_today":   surfaced_today,
        "surfaced_recent":  surfaced_recent,
        "closed_recent":    closed_recent,
        "top_factor_buckets": top_factors,
        "top_combos":       top_combos,
        "total_plays":      len(plays),
    }


def reset_weights() -> Dict:
    data = _load()
    data["weights"] = DEFAULT_WEIGHTS.copy()
    for r in data.get("regime_weights", {}):
        data["regime_weights"][r] = DEFAULT_WEIGHTS.copy()
    _save(data)
    return {"success": True, "weights": DEFAULT_WEIGHTS}
