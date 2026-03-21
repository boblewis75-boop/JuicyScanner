"""
watchlist.py
------------
Watchlist Manager for OptionsEdge AI.

Features:
  - Multiple named watchlists (e.g. "My Core", "Earnings Plays", "Tech Breakouts")
  - Add/remove symbols with optional notes and tags
  - Track when a play was added and at what underlying price
  - Simple JSON file persistence (swap to DB later if needed)
  - Alert thresholds per symbol (price, IV rank, score targets)
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional


# ─────────────────────────────────────────────────────────────────
# Storage
# ─────────────────────────────────────────────────────────────────

_LOCAL_WL = os.path.join(os.path.dirname(__file__), "..", "data", "watchlists.json")
WATCHLIST_FILE = "/app/watchlists.json" if os.path.exists("/app") else _LOCAL_WL


def _load() -> Dict:
    if os.path.exists(WATCHLIST_FILE):
        with open(WATCHLIST_FILE, "r") as f:
            return json.load(f)
    # Default structure with one starter list
    return {
        "lists": {
            "My Watchlist": {
                "id":          "my-watchlist",
                "name":        "My Watchlist",
                "description": "Default watchlist",
                "created_at":  _now(),
                "symbols":     {},
            }
        },
        "default_list": "My Watchlist",
    }


def _save(data: Dict):
    os.makedirs(os.path.dirname(WATCHLIST_FILE), exist_ok=True)
    with open(WATCHLIST_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _now() -> str:
    return datetime.now().isoformat()


# ─────────────────────────────────────────────────────────────────
# Watchlist CRUD
# ─────────────────────────────────────────────────────────────────

def get_all_lists() -> Dict[str, Any]:
    """Return all watchlists with symbol counts."""
    data = _load()
    result = {}
    for name, lst in data["lists"].items():
        result[name] = {
            "id":           lst["id"],
            "name":         lst["name"],
            "description":  lst.get("description", ""),
            "symbol_count": len(lst["symbols"]),
            "symbols":      list(lst["symbols"].keys()),
            "created_at":   lst["created_at"],
            "is_default":   name == data.get("default_list"),
        }
    return {"lists": result, "default": data.get("default_list")}


def create_list(name: str, description: str = "") -> Dict:
    """Create a new named watchlist."""
    data = _load()
    if name in data["lists"]:
        return {"success": False, "error": f"List '{name}' already exists."}

    list_id = name.lower().replace(" ", "-")
    data["lists"][name] = {
        "id":          list_id,
        "name":        name,
        "description": description,
        "created_at":  _now(),
        "symbols":     {},
    }
    _save(data)
    return {"success": True, "list": name, "message": f"Created watchlist '{name}'."}


def delete_list(name: str) -> Dict:
    data = _load()
    if name not in data["lists"]:
        return {"success": False, "error": f"List '{name}' not found."}
    if len(data["lists"]) == 1:
        return {"success": False, "error": "Cannot delete the last watchlist."}
    del data["lists"][name]
    if data.get("default_list") == name:
        data["default_list"] = list(data["lists"].keys())[0]
    _save(data)
    return {"success": True, "message": f"Deleted watchlist '{name}'."}


def set_default_list(name: str) -> Dict:
    data = _load()
    if name not in data["lists"]:
        return {"success": False, "error": f"List '{name}' not found."}
    data["default_list"] = name
    _save(data)
    return {"success": True, "message": f"'{name}' set as default watchlist."}


# ─────────────────────────────────────────────────────────────────
# Symbol management
# ─────────────────────────────────────────────────────────────────

def add_symbol(
    symbol: str,
    list_name: Optional[str]   = None,
    note: str                  = "",
    tags: Optional[List[str]]  = None,
    alert_price_above: Optional[float] = None,
    alert_price_below: Optional[float] = None,
    alert_score_above: Optional[int]   = None,
    added_price: Optional[float]       = None,
) -> Dict:
    """Add a symbol to a watchlist."""
    data = _load()
    list_name = list_name or data.get("default_list", "My Watchlist")

    if list_name not in data["lists"]:
        return {"success": False, "error": f"List '{list_name}' not found."}

    symbol = symbol.upper().strip()
    symbols = data["lists"][list_name]["symbols"]

    if symbol in symbols:
        return {"success": False, "error": f"{symbol} is already in '{list_name}'."}

    symbols[symbol] = {
        "symbol":              symbol,
        "added_at":            _now(),
        "added_price":         added_price,
        "note":                note,
        "tags":                tags or [],
        "alerts": {
            "price_above": alert_price_above,
            "price_below": alert_price_below,
            "score_above": alert_score_above,
        },
        "last_scan": None,
        "last_score": None,
    }
    _save(data)
    return {"success": True, "symbol": symbol, "list": list_name, "message": f"Added {symbol} to '{list_name}'."}


def remove_symbol(symbol: str, list_name: Optional[str] = None) -> Dict:
    data = _load()
    list_name = list_name or data.get("default_list", "My Watchlist")
    symbol    = symbol.upper().strip()

    if list_name not in data["lists"]:
        return {"success": False, "error": f"List '{list_name}' not found."}

    symbols = data["lists"][list_name]["symbols"]
    if symbol not in symbols:
        return {"success": False, "error": f"{symbol} not found in '{list_name}'."}

    del symbols[symbol]
    _save(data)
    return {"success": True, "message": f"Removed {symbol} from '{list_name}'."}


def update_symbol(
    symbol: str,
    list_name: Optional[str] = None,
    note: Optional[str]      = None,
    tags: Optional[List[str]] = None,
    alert_price_above: Optional[float] = None,
    alert_price_below: Optional[float] = None,
    alert_score_above: Optional[int]   = None,
) -> Dict:
    """Update notes, tags, or alerts for a symbol."""
    data      = _load()
    list_name = list_name or data.get("default_list", "My Watchlist")
    symbol    = symbol.upper().strip()

    if list_name not in data["lists"]:
        return {"success": False, "error": f"List '{list_name}' not found."}

    symbols = data["lists"][list_name]["symbols"]
    if symbol not in symbols:
        return {"success": False, "error": f"{symbol} not in '{list_name}'."}

    entry = symbols[symbol]
    if note is not None:
        entry["note"] = note
    if tags is not None:
        entry["tags"] = tags
    if alert_price_above is not None:
        entry["alerts"]["price_above"] = alert_price_above
    if alert_price_below is not None:
        entry["alerts"]["price_below"] = alert_price_below
    if alert_score_above is not None:
        entry["alerts"]["score_above"] = alert_score_above

    _save(data)
    return {"success": True, "message": f"Updated {symbol} in '{list_name}'."}


def update_last_scan(symbol: str, list_name: str, score: int, price: float):
    """Called after each scan to track when we last saw a play."""
    data      = _load()
    list_name = list_name or data.get("default_list", "My Watchlist")
    symbol    = symbol.upper()

    if list_name in data["lists"] and symbol in data["lists"][list_name]["symbols"]:
        data["lists"][list_name]["symbols"][symbol]["last_scan"]  = _now()
        data["lists"][list_name]["symbols"][symbol]["last_score"] = score
        data["lists"][list_name]["symbols"][symbol]["last_price"] = price
        _save(data)


def get_symbols_for_scan(list_name: Optional[str] = None) -> List[str]:
    """Get the list of symbols to scan from a watchlist."""
    data      = _load()
    list_name = list_name or data.get("default_list", "My Watchlist")
    if list_name not in data["lists"]:
        return []
    return list(data["lists"][list_name]["symbols"].keys())


def check_alerts(symbol: str, current_price: float, score: int) -> List[Dict]:
    """Check if any alerts are triggered for a symbol across all lists."""
    data    = _load()
    alerts  = []

    for list_name, lst in data["lists"].items():
        if symbol in lst["symbols"]:
            a = lst["symbols"][symbol].get("alerts", {})
            if a.get("price_above") and current_price >= a["price_above"]:
                alerts.append({
                    "type":    "PRICE_ABOVE",
                    "symbol":  symbol,
                    "list":    list_name,
                    "message": f"{symbol} crossed above ${a['price_above']:.2f} (now ${current_price:.2f})",
                })
            if a.get("price_below") and current_price <= a["price_below"]:
                alerts.append({
                    "type":    "PRICE_BELOW",
                    "symbol":  symbol,
                    "list":    list_name,
                    "message": f"{symbol} dropped below ${a['price_below']:.2f} (now ${current_price:.2f})",
                })
            if a.get("score_above") and score >= a["score_above"]:
                alerts.append({
                    "type":    "SCORE_HIT",
                    "symbol":  symbol,
                    "list":    list_name,
                    "message": f"{symbol} AI score reached {score} (threshold: {a['score_above']})",
                })

    return alerts
