#!/usr/bin/env python3
"""
spx_nodes.py — SPX GEX Node Detector using Charles Schwab API

pip install requests python-dotenv

Usage:
    python spx_nodes.py

Reads SCHWAB_APP_KEY, SCHWAB_APP_SECRET from .env
Requires a valid token.json (run schwab_auth.py once first)
"""

import os, json, time, base64
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import requests

load_dotenv()

APP_KEY    = os.getenv("SCHWAB_APP_KEY")
APP_SECRET = os.getenv("SCHWAB_APP_SECRET")
TOKEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "token.json")
API        = "https://api.schwabapi.com"


# ── Auth ───────────────────────────────────────────────────────────────────

def get_token():
    with open(TOKEN_PATH) as f:
        token = json.load(f)
    age = time.time() - token.get("creation_timestamp", 0)
    if age > token.get("expires_in", 1800) - 120:
        creds = base64.b64encode(f"{APP_KEY}:{APP_SECRET}".encode()).decode()
        r = requests.post(f"{API}/v1/oauth/token",
            headers={"Authorization": f"Basic {creds}", "Content-Type": "application/x-www-form-urlencoded"},
            data={"grant_type": "refresh_token", "refresh_token": token["refresh_token"]})
        if r.ok:
            token = r.json()
            token["creation_timestamp"] = int(time.time())
            with open(TOKEN_PATH, "w") as f: json.dump(token, f, indent=2)
    return token["access_token"]

def schwab_get(path, params=None):
    r = requests.get(f"{API}{path}",
        headers={"Authorization": f"Bearer {get_token()}", "Accept": "application/json"},
        params=params or {})
    r.raise_for_status()
    return r.json()


# ── Fetch SPX chain + spot ─────────────────────────────────────────────────

def get_spx_spot():
    d = schwab_get("/marketdata/v1/quotes", {"symbols": "$SPX"})
    q = d.get("$SPX", {}).get("quote", {})
    return q.get("lastPrice") or q.get("mark") or q.get("closePrice") or 0

def get_spx_chain(days=45):
    today = datetime.now(timezone.utc).date()
    return schwab_get("/marketdata/v1/chains", {
        "symbol":        "$SPX",
        "contractType":  "ALL",
        "fromDate":      today.isoformat(),
        "toDate":        (today + timedelta(days=days)).isoformat(),
        "includeQuotes": "TRUE",
        "strikeCount":   "120",
    })


# ── GEX computation ────────────────────────────────────────────────────────

def compute_gex(chain, spot):
    """
    GEX = sign × |gamma| × OI × 100 × spot
    sign = +1 for calls, -1 for puts
    Returns: { strike: { exp_date: gex_value } }
    """
    gex = {}
    for side, exp_map in [("CALL", chain.get("callExpDateMap", {})),
                           ("PUT",  chain.get("putExpDateMap",  {}))]:
        for exp_key, strikes in exp_map.items():
            exp_date = exp_key.split(":")[0]
            for sk, contracts in strikes.items():
                strike = round(float(sk), 2)
                for c in contracts:
                    oi    = c.get("openInterest", 0) or 0
                    gamma = c.get("gamma", 0) or 0
                    if oi == 0 or gamma == 0:
                        continue
                    sign  = 1 if side == "CALL" else -1
                    val   = sign * oi * 100 * spot * abs(gamma)
                    gex.setdefault(strike, {})
                    gex[strike][exp_date] = gex[strike].get(exp_date, 0) + val
    return gex


# ── GEX Flip Zone ──────────────────────────────────────────────────────────

def find_flip_zone(gex, strikes, spot):
    """Strike where net GEX crosses zero — closest to spot."""
    net = {s: sum(gex.get(s, {}).values()) for s in strikes}
    asc = sorted(strikes)
    crossings = []
    for i in range(len(asc) - 1):
        s1, s2 = asc[i], asc[i + 1]
        g1, g2 = net.get(s1, 0), net.get(s2, 0)
        if (g1 > 0 and g2 <= 0) or (g1 <= 0 and g2 > 0):
            if abs(g1 - g2) > 0.001:
                crossings.append(round(s1 + (s2 - s1) * (g1 / (g1 - g2)), 2))
            else:
                crossings.append(round((s1 + s2) / 2, 2))
    return min(crossings, key=lambda x: abs(x - spot)) if crossings else round(spot, 2)


# ── Node detection ─────────────────────────────────────────────────────────

def detect_nodes(gex, strikes, exps, spot):
    """
    Regime-aware GEX node labeler.
    Returns { strike: label } — one winner per zone, max ~4 labels.

    Labels:
      POS γ regime: WALL (call wall above), FLOOR (put wall below),
                    GATE (neg node above), TRAP (neg node below)
      NEG γ regime: GATE (neg above), ACCEL (neg below),
                    MGNT (pos magnet nodes)
      Special:      RUG + BREACH (compressed walls with no buffer)
    """
    net = {}
    total_net = 0
    for s in strikes:
        n = sum(gex.get(s, {}).get(e, 0) for e in exps)
        net[s] = n
        total_net += n

    mx = max((abs(v) for v in net.values()), default=1) or 1
    is_neg_regime = total_net < 0

    above = [s for s in strikes if s > spot]
    below = [s for s in strikes if s < spot]

    def strongest_pos(zone): return next(iter(sorted([s for s in zone if net[s] > 0], key=lambda s: -net[s])), None)
    def strongest_neg(zone): return next(iter(sorted([s for s in zone if net[s] < 0], key=lambda s: net[s])),  None)

    spa = strongest_pos(above)
    spb = strongest_pos(below)
    sna = strongest_neg(above)
    snb = strongest_neg(below)

    sig = lambda s: s is not None and abs(net[s]) / mx > 0.20

    nodes = {}
    if is_neg_regime:
        if sig(sna): nodes[sna] = "GATE"
        if sig(snb): nodes[snb] = "ACCEL"
        if sig(spa): nodes[spa] = "MGNT"
        if sig(spb): nodes[spb] = "MGNT"
    else:
        if sig(spa): nodes[spa] = "WALL"
        if sig(spb): nodes[spb] = "FLOOR"
        if sig(sna): nodes[sna] = "GATE"
        if sig(snb): nodes[snb] = "TRAP"

    # RUG PULL: strong positive directly above strong negative with no buffer
    if spa and snb and net[spa] / mx > 0.45 and abs(net[snb]) / mx > 0.35:
        buffer = [s for s in strikes if snb < s < spa and net[s] > mx * 0.15]
        if not buffer:
            nodes[spa] = "RUG"
            nodes[snb] = "BREACH"

    return nodes, total_net


# ── Main ───────────────────────────────────────────────────────────────────

def run():
    print("Fetching SPX spot price...")
    spot = get_spx_spot()
    if not spot:
        print("Could not get SPX spot price. Check token.json or market hours.")
        return
    print(f"SPX Spot: ${spot:,.2f}\n")

    print("Fetching SPX options chain...")
    chain = get_spx_chain(days=45)

    print("Computing GEX...")
    gex = compute_gex(chain, spot)

    # Build sorted strike list (SPX $5 increments, ±$300 of spot)
    all_strikes = sorted(gex.keys())
    strikes = [s for s in all_strikes if abs(s - spot) <= 300]

    # Use up to 5 nearest expirations
    all_exps = sorted(set(e for sd in gex.values() for e in sd.keys()))
    exps = all_exps[:5]

    # Net GEX per strike
    net = {s: sum(gex.get(s, {}).get(e, 0) for e in exps) for s in strikes}

    # Flip zone
    flip = find_flip_zone(gex, strikes, spot)

    # Nodes
    nodes, total_net = detect_nodes(gex, strikes, exps, spot)

    regime = "NEG γ (moves AMPLIFY)" if total_net < 0 else "POS γ (moves DAMPEN)"

    # Print results
    print("=" * 55)
    print(f"  SPX GEX NODES  |  Spot: ${spot:,.2f}")
    print(f"  Regime: {regime}")
    print(f"  GEX Flip Zone:  ${flip:,.2f}  ({'ABOVE' if flip > spot else 'BELOW'} spot)")
    print("=" * 55)

    for label in ["WALL", "GATE", "RUG"]:
        strikes_with_label = [(s, net[s]) for s, l in nodes.items() if l == label]
        for s, val in sorted(strikes_with_label, reverse=True):
            print(f"  {label:<8}  ${s:>7,.2f}  GEX: {val/1e6:+.2f}M")

    print(f"  {'SPOT':<8}  ${spot:>7,.2f}  ← current")

    for label in ["FLOOR", "ACCEL", "MGNT", "TRAP", "BREACH"]:
        strikes_with_label = [(s, net[s]) for s, l in nodes.items() if l == label]
        for s, val in sorted(strikes_with_label, reverse=True):
            print(f"  {label:<8}  ${s:>7,.2f}  GEX: {val/1e6:+.2f}M")

    print("=" * 55)
    print(f"\nNearby strikes (±$50 of spot):")
    nearby = sorted([s for s in strikes if abs(s - spot) <= 50], reverse=True)
    for s in nearby:
        val = net.get(s, 0)
        lbl = nodes.get(s, "")
        flip_marker = " ← FLIP" if abs(s - flip) < 5 else ""
        spot_marker = " ← SPOT" if abs(s - spot) < 3 else ""
        bar_len = int(min(abs(val) / max(abs(v) for v in net.values() if v) * 20, 20))
        bar = ("█" if val > 0 else "░") * bar_len
        print(f"  {s:>7,.0f}  {val/1e6:>+7.2f}M  {bar:<20}  {lbl:<8}{flip_marker}{spot_marker}")


if __name__ == "__main__":
    run()
