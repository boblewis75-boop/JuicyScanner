"""
probability.py
--------------
Play Likelihood Engine for OptionsEdge AI.

Calculates the probability that an options play will hit its target using
multiple models and combines them into a single "Hit Likelihood %" score.

Models used:
  1. Delta-based probability     — market's implied probability of expiring ITM
  2. Black-Scholes probability   — prob of underlying reaching target price
  3. Technical confirmation score — how strongly technicals support the direction
  4. Historical move probability  — based on IV and DTE, what % of the time
                                    does the underlying move this far in this window?

Final score = weighted blend of all four models.
"""

import math
from typing import Optional


# ─────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────

def calculate_likelihood(
    option_type: str,          # "CALL" or "PUT"
    underlying_price: float,
    strike: float,
    premium: float,
    iv: float,                 # implied volatility as decimal (e.g. 0.45 = 45%)
    dte: int,                  # days to expiration
    delta: float,              # option delta (always positive here)
    target_pct: float,         # % move needed in underlying (always positive)
    tech_score: Optional[float] = None,  # 0-100 from technical analysis
) -> dict:
    """
    Returns a dict with:
      - likelihood_pct     : overall probability 0-100
      - delta_prob         : market's implied probability (delta × 100)
      - bs_prob            : Black-Scholes prob of hitting target
      - historical_prob    : historical move probability from IV/DTE
      - tech_confirmation  : technical analysis confirmation 0-100
      - confidence         : HIGH / MEDIUM / LOW
      - breakdown          : human-readable explanation
    """
    t = dte / 365.0  # time in years

    # ── Model 1: Delta probability ──────────────────────────────
    # Delta ≈ probability of expiring in the money (rough but fast)
    delta_prob = abs(delta) * 100

    # ── Model 2: Black-Scholes probability of hitting target ────
    # Prob that S reaches target_price before expiry (simplified)
    target_price = underlying_price * (1 + target_pct / 100)
    bs_prob = _bs_touch_probability(
        S=underlying_price,
        target=target_price,
        iv=iv,
        t=t,
        direction=option_type,
    )

    # ── Model 3: Historical move probability ────────────────────
    # Given IV and DTE, what % of the time does a stock move this far?
    hist_prob = _historical_move_probability(
        target_pct=target_pct,
        iv=iv,
        dte=dte,
    )

    # ── Model 4: Technical confirmation ─────────────────────────
    # If we have a tech score, normalize it; otherwise neutral 50
    tech_conf = tech_score if tech_score is not None else 50.0

    # ── Weighted blend ───────────────────────────────────────────
    # Delta is most reliable for ITM prob, BS for target touch,
    # historical adds base-rate anchor, tech adds directional edge
    weights = {
        "delta":    0.25,
        "bs":       0.30,
        "hist":     0.25,
        "tech":     0.20,
    }

    likelihood = (
        delta_prob    * weights["delta"] +
        bs_prob       * weights["bs"]    +
        hist_prob     * weights["hist"]  +
        tech_conf     * weights["tech"]
    )
    likelihood = round(min(95, max(5, likelihood)), 1)

    # ── Confidence tier ──────────────────────────────────────────
    if likelihood >= 65:
        confidence = "HIGH"
    elif likelihood >= 45:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # ── Human-readable breakdown ─────────────────────────────────
    breakdown = _build_breakdown(
        option_type, delta_prob, bs_prob, hist_prob,
        tech_conf, likelihood, target_pct, dte
    )

    return {
        "likelihood_pct":    likelihood,
        "delta_prob":        round(delta_prob, 1),
        "bs_prob":           round(bs_prob, 1),
        "historical_prob":   round(hist_prob, 1),
        "tech_confirmation": round(tech_conf, 1),
        "confidence":        confidence,
        "breakdown":         breakdown,
    }


# ─────────────────────────────────────────────────────────────────
# Model implementations
# ─────────────────────────────────────────────────────────────────

def _bs_touch_probability(
    S: float,
    target: float,
    iv: float,
    t: float,
    direction: str,
) -> float:
    """
    Probability that price touches 'target' at any point before expiry.
    Uses the reflection principle approximation for barrier probability.

    P(touch) = N(d2) + exp(2 * ln(S/K) / (iv^2 * t)) * N(d1)
    where d1, d2 are adjusted Black-Scholes terms.
    """
    if iv <= 0 or t <= 0 or S <= 0 or target <= 0:
        return 50.0

    log_ratio = math.log(S / target)
    vol_sqrt_t = iv * math.sqrt(t)

    if vol_sqrt_t < 1e-10:
        return 50.0

    d1 = (-log_ratio + 0.5 * iv**2 * t) / vol_sqrt_t
    d2 = (-log_ratio - 0.5 * iv**2 * t) / vol_sqrt_t

    # Reflection principle barrier probability
    lam = 2 * log_ratio / (iv**2 * t)
    prob = _norm_cdf(d2) + math.exp(lam) * _norm_cdf(d1)

    if direction == "PUT":
        prob = 1 - prob

    return round(min(95, max(5, prob * 100)), 1)


def _historical_move_probability(
    target_pct: float,
    iv: float,
    dte: int,
) -> float:
    """
    Based on a lognormal distribution of returns (implied by IV),
    what is the probability the underlying moves at least target_pct%
    in 'dte' days?

    This is the complementary CDF of a normal distribution.
    """
    if iv <= 0 or dte <= 0:
        return 50.0

    t = dte / 365.0
    # Expected 1-sigma move over the period
    expected_move_pct = iv * math.sqrt(t) * 100

    # Z-score: how many sigmas is the target move?
    if expected_move_pct < 0.01:
        return 50.0

    z = target_pct / expected_move_pct

    # P(move >= target) = 1 - N(z)   (one-tailed)
    prob = (1 - _norm_cdf(z)) * 100

    return round(min(92, max(5, prob)), 1)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF using math.erf approximation."""
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


# ─────────────────────────────────────────────────────────────────
# Breakdown text
# ─────────────────────────────────────────────────────────────────

def _build_breakdown(
    opt_type, delta_prob, bs_prob, hist_prob,
    tech_conf, likelihood, target_pct, dte
) -> str:
    direction = "upward" if opt_type == "CALL" else "downward"
    tech_note = (
        "Technical signals strongly confirm the direction."
        if tech_conf >= 70
        else "Technical signals are mixed — proceed with caution."
        if tech_conf < 45
        else "Technicals offer moderate confirmation."
    )
    hist_note = (
        f"Historically, the underlying makes a {target_pct:.0f}%+ {direction} move in {dte} days roughly {hist_prob:.0f}% of the time based on current IV."
    )
    return (
        f"Delta model: {delta_prob:.0f}% ITM probability. "
        f"B-S barrier model: {bs_prob:.0f}% chance of touching target. "
        f"{hist_note} "
        f"{tech_note} "
        f"Combined likelihood: {likelihood}%."
    )
