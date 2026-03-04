"""
server.py
---------
FastAPI REST server for OptionsEdge AI.

Endpoints:
  GET  /scan                      — Scan full watchlist
  GET  /scan/{symbol}             — Scan single ticker
  GET  /technicals/{symbol}       — Full technical analysis for a symbol
  GET  /quote/{symbol}            — Current quote
  GET  /watchlist                 — Get all watchlists
  POST /watchlist                 — Create new watchlist
  DELETE /watchlist/{name}        — Delete a watchlist
  GET  /watchlist/{name}/symbols  — Get symbols in a list
  POST /watchlist/{name}/symbols  — Add symbol to list
  DELETE /watchlist/{name}/symbols/{symbol} — Remove symbol
  PUT  /watchlist/{name}/symbols/{symbol}   — Update symbol (note/tags/alerts)
  GET  /watchlist/{name}/scan     — Scan a specific watchlist
  GET  /auth/url                  — Schwab OAuth URL
  POST /auth/token                — Exchange auth code
  GET  /health                    — Health check
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

from config import (
    USE_LIVE_DATA, WATCHLIST,
    DEFAULT_MIN_RR, DEFAULT_MIN_TARGET, DEFAULT_MAX_DTE,
    DEFAULT_MIN_SCORE, DEFAULT_MIN_OI, DEFAULT_MAX_SPREAD,
    HOST, PORT
)
from core.scanner import scan_options
from core.technicals import analyze as analyze_technicals, generate_mock_bars
from core import watchlist as wl
from data.mock_data import get_mock_options_chain, get_mock_quote


# ----------------------------------------------------------------
app = FastAPI(
    title="OptionsEdge AI",
    description="AI-powered options scanner with probability, technicals, and watchlist management",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_schwab_client = None

def get_schwab_client():
    global _schwab_client
    if _schwab_client is None:
        from api.schwab_client import SchwabClient
        _schwab_client = SchwabClient()
    return _schwab_client


def fetch_chain(symbol: str, max_dte: int = DEFAULT_MAX_DTE) -> dict:
    if USE_LIVE_DATA:
        return get_schwab_client().get_options_chain(symbol, days_to_exp=max_dte)
    return get_mock_options_chain(symbol)


def fetch_quote(symbol: str) -> dict:
    if USE_LIVE_DATA:
        return get_schwab_client().get_quote(symbol)
    return get_mock_quote(symbol)


def fetch_bars(symbol: str) -> list:
    """Fetch OHLCV daily bars — real Schwab data in live mode, converted to Bar objects."""
    from core.technicals import Bar
    if USE_LIVE_DATA:
        try:
            raw = get_schwab_client().get_price_history(symbol, period_type="month", period=6)
            if not raw or len(raw) < 10:
                raise ValueError(f"Too few bars: {len(raw) if raw else 0}")
            # Convert Schwab dicts → Bar objects
            bars = []
            for r in raw:
                bars.append(Bar(
                    date=str(r.get("datetime", "")),
                    open=float(r.get("open", 0)),
                    high=float(r.get("high", 0)),
                    low=float(r.get("low", 0)),
                    close=float(r.get("close", 0)),
                    volume=int(r.get("volume", 0)),
                ))
            return bars
        except Exception as e:
            print(f"⚠️  Price history failed for {symbol}: {e}, falling back to mock")
            return generate_mock_bars(symbol)
    return generate_mock_bars(symbol)


# ----------------------------------------------------------------
# Health
# ----------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode":   "LIVE" if USE_LIVE_DATA else "MOCK",
        "version": "2.0.0",
        "features": ["options_scanner", "probability_engine", "technical_analysis", "watchlist_manager"],
    }


# ----------------------------------------------------------------
# Scan
# ----------------------------------------------------------------

@app.get("/scan")
def scan_all(
    min_rr:        float = Query(DEFAULT_MIN_RR),
    min_target:    float = Query(DEFAULT_MIN_TARGET),
    max_dte:       int   = Query(DEFAULT_MAX_DTE),
    min_score:     int   = Query(DEFAULT_MIN_SCORE),
    min_oi:        int   = Query(50),
    contract_type: str   = Query("ALL"),
    max_premium:   float = Query(999,  description="Max option premium price"),
    exclude:       str   = Query(None, description="Comma-separated symbols to exclude"),
    dedup:         bool  = Query(True,  description="Show only top play per ticker"),
    symbols:       Optional[str] = Query(None, description="Comma-separated symbols"),
):
    tickers     = symbols.upper().split(",") if symbols else WATCHLIST
    all_results = []
    errors      = []

    def _scan_one(symbol):
        try:
            chain = fetch_chain(symbol, max_dte)
            quote = fetch_quote(symbol)
            # Skip price history during bulk scan — too many API calls
            # Scanner uses technicals only for direction bias, mock bars are fine
            bars = generate_mock_bars(symbol)
            plays = scan_options(
                chain, quote, bars=bars,
                min_rr=min_rr, min_target=min_target, max_dte=max_dte,
                min_score=min_score, min_oi=min_oi, contract_type=contract_type,
            )
            return plays, None
        except Exception as e:
            return [], {"symbol": symbol, "error": str(e)}

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_scan_one, sym): sym for sym in tickers}
        for future in as_completed(futures, timeout=45):
            try:
                plays, err = future.result()
                if err:
                    errors.append(err)
                else:
                    all_results.extend(plays)
            except Exception as e:
                errors.append({"symbol": "unknown", "error": str(e)})

    all_results.sort(key=lambda x: x["score"], reverse=True)

    # Apply premium filter
    if max_premium < 999:
        all_results = [p for p in all_results if p.get("premium", 999) <= max_premium]

    # Apply exclude filter (indexes etc)
    if exclude:
        excl_set = {s.strip().upper() for s in exclude.split(",")}
        all_results = [p for p in all_results if p.get("ticker", "").upper() not in excl_set]

    # Deduplication — keep only top-scoring play per ticker
    if dedup:
        seen = {}
        deduped = []
        for p in all_results:  # already sorted by score desc
            ticker = p.get("ticker", "")
            if ticker not in seen:
                seen[ticker] = True
                deduped.append(p)
        all_results = deduped

    return {
        "mode":    "LIVE" if USE_LIVE_DATA else "MOCK",
        "scanned": len(tickers),
        "found":   len(all_results),
        "plays":   all_results[:50],
        "errors":  errors,
        "filters": {
            "min_rr": min_rr, "min_target": min_target, "max_dte": max_dte,
            "min_score": min_score, "contract_type": contract_type,
            "max_premium": max_premium,
        }
    }


@app.get("/scan/{symbol}")
def scan_single(
    symbol:        str,
    min_rr:        float = Query(DEFAULT_MIN_RR),
    min_target:    float = Query(DEFAULT_MIN_TARGET),
    max_dte:       int   = Query(DEFAULT_MAX_DTE),
    min_score:     int   = Query(0),
    contract_type: str   = Query("ALL"),
):
    symbol = symbol.upper()
    try:
        chain = fetch_chain(symbol, max_dte)
        quote = fetch_quote(symbol)
        bars  = fetch_bars(symbol)
        plays = scan_options(
            chain, quote, bars=bars,
            min_rr=min_rr, min_target=min_target, max_dte=max_dte,
            min_score=min_score, contract_type=contract_type,
        )
        return {
            "symbol":          symbol,
            "underlyingPrice": chain["underlyingPrice"],
            "mode":            "LIVE" if USE_LIVE_DATA else "MOCK",
            "found":           len(plays),
            "plays":           plays,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------------------
# Technical Analysis
# ----------------------------------------------------------------

@app.get("/technicals/{symbol}")
def get_technicals(symbol: str):
    """Full technical analysis for a symbol — RSI, MACD, MAs, volume, channel."""
    symbol = symbol.upper()
    try:
        # Try real bars first, fall back to mock on any error
        try:
            bars = fetch_bars(symbol)
            if not bars or len(bars) < 10:
                raise ValueError(f"Not enough bars: {len(bars) if bars else 0}")
        except Exception as bar_err:
            print(f"⚠️  Bars failed for {symbol}: {bar_err}, using mock")
            bars = generate_mock_bars(symbol)

        tech = analyze_technicals(bars, symbol=symbol)

        # Try to get live price, fail silently
        try:
            quote = fetch_quote(symbol)
            live_price = (
                quote.get("lastPrice") or
                quote.get("mark") or
                quote.get("last") or
                quote.get("closePrice") or 0
            )
            if live_price:
                tech["current_price"] = live_price
                tech["live_price"]    = live_price
        except Exception as q_err:
            print(f"⚠️  Quote failed for {symbol}: {q_err}")

        return tech
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------------------
# Quote
# ----------------------------------------------------------------

@app.get("/quote/{symbol}")
def get_quote(symbol: str):
    symbol = symbol.upper()
    try:
        return fetch_quote(symbol)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------------------
# Watchlist endpoints
# ----------------------------------------------------------------

@app.get("/watchlist")
def get_all_watchlists():
    """Get all watchlists."""
    return wl.get_all_lists()


class CreateListRequest(BaseModel):
    name: str
    description: str = ""

@app.post("/watchlist")
def create_watchlist(req: CreateListRequest):
    result = wl.create_list(req.name, req.description)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.delete("/watchlist/{name}")
def delete_watchlist(name: str):
    result = wl.delete_list(name)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/watchlist/{name}/default")
def set_default(name: str):
    result = wl.set_default_list(name)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


class AddSymbolRequest(BaseModel):
    symbol: str
    note: str = ""
    tags: List[str] = []
    added_price: Optional[float] = None
    alert_price_above: Optional[float] = None
    alert_price_below: Optional[float] = None
    alert_score_above: Optional[int]   = None

@app.post("/watchlist/{name}/symbols")
def add_to_watchlist(name: str, req: AddSymbolRequest):
    result = wl.add_symbol(
        symbol=req.symbol, list_name=name, note=req.note, tags=req.tags,
        added_price=req.added_price,
        alert_price_above=req.alert_price_above,
        alert_price_below=req.alert_price_below,
        alert_score_above=req.alert_score_above,
    )
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.delete("/watchlist/{name}/symbols/{symbol}")
def remove_from_watchlist(name: str, symbol: str):
    result = wl.remove_symbol(symbol, name)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


class UpdateSymbolRequest(BaseModel):
    note: Optional[str]            = None
    tags: Optional[List[str]]      = None
    alert_price_above: Optional[float] = None
    alert_price_below: Optional[float] = None
    alert_score_above: Optional[int]   = None

@app.put("/watchlist/{name}/symbols/{symbol}")
def update_watchlist_symbol(name: str, symbol: str, req: UpdateSymbolRequest):
    result = wl.update_symbol(
        symbol=symbol, list_name=name, note=req.note, tags=req.tags,
        alert_price_above=req.alert_price_above,
        alert_price_below=req.alert_price_below,
        alert_score_above=req.alert_score_above,
    )
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.get("/watchlist/{name}/scan")
def scan_watchlist(
    name:          str,
    min_rr:        float = Query(DEFAULT_MIN_RR),
    min_target:    float = Query(DEFAULT_MIN_TARGET),
    max_dte:       int   = Query(DEFAULT_MAX_DTE),
    min_score:     int   = Query(DEFAULT_MIN_SCORE),
    contract_type: str   = Query("ALL"),
):
    """Scan all symbols in a specific watchlist."""
    tickers = wl.get_symbols_for_scan(name)
    if not tickers:
        return {"found": 0, "plays": [], "message": f"No symbols in watchlist '{name}'"}

    all_results = []
    errors = []
    triggered_alerts = []

    for symbol in tickers:
        try:
            chain = fetch_chain(symbol, max_dte)
            quote = fetch_quote(symbol)
            bars  = fetch_bars(symbol)
            plays = scan_options(
                chain, quote, bars=bars,
                min_rr=min_rr, min_target=min_target, max_dte=max_dte,
                min_score=min_score, contract_type=contract_type,
            )
            # Update last scan info
            top_score = plays[0]["score"] if plays else 0
            wl.update_last_scan(symbol, name, top_score, chain["underlyingPrice"])
            # Check alerts
            alerts = wl.check_alerts(symbol, chain["underlyingPrice"], top_score)
            triggered_alerts.extend(alerts)
            all_results.extend(plays)
        except Exception as e:
            errors.append({"symbol": symbol, "error": str(e)})

    all_results.sort(key=lambda x: x["score"], reverse=True)
    return {
        "watchlist":        name,
        "scanned":          len(tickers),
        "found":            len(all_results),
        "plays":            all_results[:50],
        "triggered_alerts": triggered_alerts,
        "errors":           errors,
    }


# ----------------------------------------------------------------
# Auth
# ----------------------------------------------------------------

@app.get("/auth/url")
def get_auth_url():
    """Returns the Schwab auth URL — open it in your browser to authorize."""
    client = get_schwab_client()
    url = client.get_auth_url()
    # Redirect directly to Schwab login
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url=url)


@app.get("/auth/callback")
def auth_callback(code: str = None, session: str = None, error: str = None):
    """
    Schwab redirects here after authorization.
    Auto-exchanges the code for tokens — no manual step needed.
    Just open /auth/url and approve, tokens are saved automatically.
    """
    from fastapi.responses import HTMLResponse
    if error:
        return HTMLResponse(f"""
        <html><body style="font-family:monospace;background:#0d0910;color:#f87171;padding:40px">
        <h2>❌ Auth Error</h2><p>{error}</p>
        </body></html>""")
    if not code:
        return HTMLResponse("""
        <html><body style="font-family:monospace;background:#0d0910;color:#f87171;padding:40px">
        <h2>❌ No code received</h2>
        </body></html>""")
    try:
        get_schwab_client().exchange_code_for_tokens(code)
        return HTMLResponse("""
        <html><body style="font-family:monospace;background:#0d0910;color:#a855f7;padding:40px;text-align:center">
        <h1 style="font-size:3rem">✅</h1>
        <h2 style="color:#22c55e">Authenticated Successfully!</h2>
        <p style="color:#9ca3af">Tokens saved. JuicyScanner is now in LIVE MODE.</p>
        <p style="color:#9ca3af;margin-top:20px">You can close this tab.</p>
        <script>setTimeout(()=>window.close(),3000)</script>
        </body></html>""")
    except Exception as e:
        return HTMLResponse(f"""
        <html><body style="font-family:monospace;background:#0d0910;color:#f87171;padding:40px">
        <h2>❌ Token Exchange Failed</h2><pre>{str(e)}</pre>
        </body></html>""")


class TokenRequest(BaseModel):
    code: str

@app.post("/auth/token")
def exchange_token(req: TokenRequest):
    """Manual token exchange (fallback)."""
    try:
        # Handle both raw code and full URL
        code = req.code
        if "session" in code:
            from urllib.parse import parse_qs, urlparse
            qs = parse_qs(urlparse("?" + code.split("?")[-1]).query)
            code = qs.get("code", [code])[0]
        get_schwab_client().exchange_code_for_tokens(code)
        return {"status": "success", "message": "Authenticated with Schwab successfully."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ----------------------------------------------------------------
# Run
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# DEBUG endpoint — see raw Schwab data
# ----------------------------------------------------------------

@app.get("/debug/{symbol}")
async def debug_chain(symbol: str):
    """Returns raw chain structure so we can see what Schwab actually sends."""
    try:
        chain = fetch_chain(symbol.upper(), max_dte=45)
        
        # Sample first few contracts from each side
        call_map = chain.get("callExpDateMap", {})
        put_map  = chain.get("putExpDateMap",  {})
        
        # Get first expiry
        first_call_exp = list(call_map.keys())[:2] if call_map else []
        first_put_exp  = list(put_map.keys())[:2]  if put_map  else []
        
        # Sample contracts
        sample_calls = {}
        for exp in first_call_exp:
            val = call_map[exp]
            sample_calls[exp] = {
                "type": type(val).__name__,
                "len": len(val),
                "sample": str(val)[:500] if not isinstance(val, dict) else {k: str(v)[:200] for k,v in list(val.items())[:2]}
            }
        
        sample_puts = {}
        for exp in first_put_exp:
            val = put_map[exp]
            sample_puts[exp] = {
                "type": type(val).__name__,
                "len": len(val),
                "sample": str(val)[:500] if not isinstance(val, dict) else {k: str(v)[:200] for k,v in list(val.items())[:2]}
            }

        # Try flatten
        from core.scanner import _flatten_chain
        contracts = _flatten_chain(chain, "ALL")
        sample_contract = contracts[0] if contracts else {}

        return {
            "symbol": symbol,
            "underlying_price": chain.get("underlyingPrice"),
            "status": chain.get("status"),
            "mode": "LIVE" if USE_LIVE_DATA else "MOCK",
            "call_expiries_count": len(call_map),
            "put_expiries_count": len(put_map),
            "first_call_expiries": first_call_exp,
            "sample_calls": sample_calls,
            "sample_puts": sample_puts,
            "flattened_contracts_count": len(contracts),
            "sample_contract": sample_contract,
            "chain_keys": list(chain.keys()),
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}

@app.get("/debug/scan/{symbol}")
async def debug_scan(symbol: str):
    """Shows exactly how many contracts pass each filter stage."""
    try:
        chain = fetch_chain(symbol.upper(), max_dte=45)
        spot  = chain.get("underlyingPrice", 0)
        
        from core.scanner import _flatten_chain
        import math
        
        all_contracts = _flatten_chain(chain, "ALL")
        
        stats = {
            "total_flattened": len(all_contracts),
            "spot": spot,
            "mode": "LIVE" if USE_LIVE_DATA else "MOCK",
        }
        
        # Stage by stage filter counts
        after_dte       = [c for c in all_contracts if c.get("daysToExpiration", 0) <= 45]
        after_oi        = [c for c in after_dte      if c.get("openInterest", 0) >= 50]
        after_bid       = [c for c in after_oi       if (c.get("bid", 0) or 0) > 0]
        after_nonstand  = [c for c in after_bid      if not c.get("nonStandard", False)]
        
        def spread_ok(c):
            bid = c.get("bid", 0) or 0
            ask = c.get("ask", 0) or 0
            mid = (bid + ask) / 2
            if mid <= 0: return False
            return (ask - bid) / mid <= 0.35
        
        after_spread = [c for c in after_nonstand if spread_ok(c)]
        
        # Sample a passing contract
        sample = after_spread[0] if after_spread else {}
        sample_fields = {k: sample.get(k) for k in 
            ["putCall","strikePrice","bid","ask","delta","gamma","theta",
             "volatility","openInterest","totalVolume","daysToExpiration",
             "nonStandard","inTheMoney"]} if sample else {}
        
        stats.update({
            "after_dte_filter":        len(after_dte),
            "after_oi_filter":         len(after_oi),
            "after_bid_filter":        len(after_bid),
            "after_nonstandard_filter":len(after_nonstand),
            "after_spread_filter":     len(after_spread),
            "sample_passing_contract": sample_fields,
        })
        
        # Check target_pct on passing contracts
        if after_spread:
            targets = []
            for c in after_spread[:20]:
                bid = c.get("bid",0) or 0
                ask = c.get("ask",0) or 0
                mid = (bid+ask)/2
                delta = abs(c.get("delta",0) or 0)
                if delta > 0.01 and mid > 0:
                    move = mid / delta
                    pct  = (move / spot) * 100 if spot else 0
                    targets.append(round(pct, 2))
            stats["sample_target_pcts"] = targets
            stats["contracts_with_target_lt_15pct"] = sum(1 for t in targets if t < 15)
        
        return stats
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("  OptionsEdge AI v2.0 — Backend")
    print(f"  Mode:     {'🔴 LIVE (Schwab)' if USE_LIVE_DATA else '🟡 MOCK (no key needed)'}")
    print(f"  Features: Scanner · Probability · Technicals · Watchlists")
    print(f"  Docs:     http://localhost:{PORT}/docs")
    print(f"  Scan:     http://localhost:{PORT}/scan")
    print(f"  Tech:     http://localhost:{PORT}/technicals/NVDA")
    print("="*60 + "\n")
    uvicorn.run("server:app", host=HOST, port=PORT, reload=True)


# ----------------------------------------------------------------
# AI Learning endpoints
# ----------------------------------------------------------------

from core.ai_learning import (
    log_play as _log_play,
    close_play as _close_play,
    get_state as _get_learning_state,
    get_open_plays as _get_open_plays,
    reset_weights as _reset_weights,
)

@app.get("/learning")
def get_learning():
    """Full AI learning state — weights, stats, insights."""
    return _get_learning_state()

@app.get("/learning/plays")
def get_open_plays():
    """All open (not yet closed) logged plays."""
    return {"plays": _get_open_plays()}

class LogPlayRequest(BaseModel):
    ticker: str
    option_type: str
    strike: float
    expiry: str
    premium: float
    score: int
    rr: float
    likelihood_pct: float
    tech_bias: float
    iv_rank: int
    delta: float
    target_pct: float
    dte: int
    tech_signals: List[str] = []
    notes: str = ""

@app.post("/learning/plays")
def log_play_endpoint(req: LogPlayRequest):
    """Log a new play when you enter a trade."""
    result = _log_play(**req.dict())
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error"))
    return result

class ClosePlayRequest(BaseModel):
    exit_premium: float
    result: str   # WIN, LOSS, PARTIAL, EXPIRED
    notes: str = ""

@app.post("/learning/plays/{play_id}/close")
def close_play_endpoint(play_id: str, req: ClosePlayRequest):
    """Close a play with outcome — triggers AI weight adjustment."""
    result = _close_play(play_id, req.exit_premium, req.result, req.notes)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error"))
    return result

class UpdatePlayRequest(BaseModel):
    premium: float = None
    notes: str     = None

@app.patch("/learning/plays/{play_id}")
def update_play_endpoint(play_id: str, req: UpdatePlayRequest):
    """Update entry price or notes on an open play."""
    from core.ai_learning import _load, _save
    data = _load()
    play = next((p for p in data["plays"] if p["id"] == play_id), None)
    if not play:
        raise HTTPException(status_code=404, detail="Play not found")
    if req.premium is not None:
        play["premium"] = round(req.premium, 2)
    if req.notes is not None:
        play["notes"] = req.notes
    _save(data)
    return {"success": True, "play": play}

@app.post("/learning/reset")
def reset_learning():
    """Reset AI weights back to defaults."""
    return _reset_weights()


# ----------------------------------------------------------------
# 0DTE endpoints — real-time polling + WebSocket stream
# ----------------------------------------------------------------

import asyncio
import json as json_lib
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from core.zero_dte import (
    scan_0dte,
    get_session_window,
    minutes_until_close,
    generate_mock_0dte_chain,
    WINDOWS,
)

# 0DTE watchlist — most liquid 0DTE underlyings
ZERO_DTE_WATCHLIST = ["SPY", "QQQ", "AAPL", "NVDA", "TSLA", "META", "AMZN", "MSFT"]


def fetch_0dte_chain(symbol: str) -> dict:
    if USE_LIVE_DATA:
        # Schwab: get_options_chain with DTE=0 filter
        return get_schwab_client().get_options_chain(symbol, days_to_exp=0, contract_type="ALL")
    return generate_mock_0dte_chain(symbol)


async def run_0dte_scan_async(
    min_score: int       = 65,
    min_likelihood: float = 55.0,
    contract_type: str   = "ALL",
    symbols: list        = None,
) -> dict:
    """Run a full 0DTE scan — called by both REST and WebSocket."""
    tickers  = symbols or ZERO_DTE_WATCHLIST
    window   = get_session_window()
    mins     = minutes_until_close()
    all_plays = []
    errors   = []

    for symbol in tickers:
        try:
            chain = fetch_0dte_chain(symbol)
            quote = fetch_quote(symbol)
            bars  = fetch_bars(symbol)
            tech  = analyze_technicals(bars, symbol=symbol)

            plays = scan_0dte(
                chain=chain, quote=quote, tech=tech,
                min_score=min_score,
                min_likelihood=min_likelihood,
                contract_type=contract_type,
            )
            all_plays.extend(plays)
        except Exception as e:
            errors.append({"symbol": symbol, "error": str(e)})

    all_plays.sort(key=lambda x: x["score"], reverse=True)

    return {
        "timestamp":    datetime.now().isoformat(),
        "mode":         "LIVE" if USE_LIVE_DATA else "MOCK",
        "window":       window,
        "minutes_left": mins,
        "market_open":  window["quality"] not in ("CLOSED",),
        "found":        len(all_plays),
        "plays":        all_plays[:10],
        "errors":       errors,
        "filters": {
            "min_score":      min_score,
            "min_likelihood": min_likelihood,
            "contract_type":  contract_type,
        }
    }


@app.get("/zero-dte")
async def scan_zero_dte(
    min_score:      int   = Query(65,   description="Minimum AI score"),
    min_likelihood: float = Query(55.0, description="Minimum hit likelihood %"),
    contract_type:  str   = Query("ALL"),
    symbols: Optional[str] = Query(None, description="Comma-separated symbols"),
):
    """Single 0DTE scan — returns current high-probability plays."""
    sym_list = symbols.upper().split(",") if symbols else None
    return await run_0dte_scan_async(min_score, min_likelihood, contract_type, sym_list)


@app.get("/zero-dte/session")
def get_session():
    """Current market session window and timing info."""
    window = get_session_window()
    mins   = minutes_until_close()
    now    = datetime.now()
    return {
        "window":       window,
        "minutes_left": mins,
        "time_et":      now.strftime("%H:%M:%S"),
        "market_open":  window["quality"] not in ("CLOSED",),
        "all_windows":  WINDOWS,
    }


# ── WebSocket — pushes updated 0DTE plays every N seconds ────────

class ZeroDteConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(json_lib.dumps(data))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


ws_manager = ZeroDteConnectionManager()


@app.websocket("/zero-dte/stream")
async def zero_dte_stream(
    websocket: WebSocket,
    interval:  int   = 30,         # refresh every N seconds
    min_score: int   = 65,
    min_likelihood: float = 55.0,
):
    """
    WebSocket stream. Sends updated 0DTE plays every `interval` seconds.
    Client connects once, receives live updates automatically.
    
    Message format: { timestamp, window, plays, found, minutes_left }
    """
    await ws_manager.connect(websocket)
    try:
        while True:
            data = await run_0dte_scan_async(min_score, min_likelihood)
            await websocket.send_text(json_lib.dumps(data))

            # Check if market is still open
            if data["minutes_left"] <= 0:
                await websocket.send_text(json_lib.dumps({
                    "type": "MARKET_CLOSED",
                    "message": "Market closed. 0DTE stream paused."
                }))
                break

            await asyncio.sleep(interval)
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        ws_manager.disconnect(websocket)


# ── SSE fallback (for browsers that prefer EventSource) ──────────

@app.get("/zero-dte/sse")
async def zero_dte_sse(
    interval:       int   = Query(30),
    min_score:      int   = Query(65),
    min_likelihood: float = Query(55.0),
):
    """
    Server-Sent Events stream — alternative to WebSocket.
    Use EventSource in the browser: new EventSource('/zero-dte/sse')
    """
    async def event_generator():
        while True:
            data  = await run_0dte_scan_async(min_score, min_likelihood)
            yield f"data: {json_lib.dumps(data)}\n\n"
            if data["minutes_left"] <= 0:
                yield "data: {\"type\": \"MARKET_CLOSED\"}\n\n"
                break
            await asyncio.sleep(interval)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Serve the frontend HTML directly from the backend ────────────
# This means your Railway URL IS your app — no separate hosting needed.

from fastapi.responses import HTMLResponse
import pathlib

@app.get("/app", response_class=HTMLResponse)
@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    """Serve juicyscanner.html from the same server."""
    html_path = pathlib.Path(__file__).parent / "juicyscanner.html"
    if html_path.exists():
        content = html_path.read_text()
        # Auto-inject the correct API URL so the frontend always points to itself
        content = content.replace(
            "const API = 'http://localhost:8000'",
            "const API = ''"   # empty string = same origin, works on any domain
        )
        return HTMLResponse(content=content)
    return HTMLResponse("<h1>juicyscanner.html not found — add it to the project folder</h1>", status_code=404)


# ----------------------------------------------------------------
# GEX (Gamma Exposure) Heatmap endpoint
# ----------------------------------------------------------------

import math as _math

@app.get("/gex/{symbol}")
def get_gex(
    symbol:       str,
    days:         int = Query(5,  description="Number of expiry dates to include"),
    strike_range: int = Query(30, description="Number of strikes above/below ATM"),
):
    """GEX = gamma x open_interest x 100 x spot^2"""
    sym = symbol.upper()

    if USE_LIVE_DATA:
        client = get_schwab_client()
        chain  = client.get_options_chain(sym, days_to_exp=days*5, contract_type="ALL")
        spot   = chain.get("underlyingPrice") or 0
        if spot <= 0:
            try:
                q    = client.get_quote(sym)
                spot = q.get("lastPrice") or q.get("mark") or q.get("closePrice") or 0
            except Exception:
                spot = 0
        if spot <= 0:
            try:
                first_exp    = next(iter(chain.get("callExpDateMap", {}).values()))
                first_strike = float(next(iter(first_exp.keys())))
                spot = first_strike
            except Exception:
                spot = 100
    else:
        chain = _generate_mock_gex_chain(symbol, days, strike_range)
        spot  = chain["underlyingPrice"]
        sym   = symbol.upper()

    # Collect all expiry dates
    all_expiries = set()
    call_map = {}  # {expiry: {strike: gex}}
    put_map  = {}

    def _parse_contracts(exp_map, is_call):
        for exp_key, options in exp_map.items():
            expiry = exp_key.split(":")[0] if ":" in exp_key else exp_key
            if expiry not in (call_map if is_call else put_map):
                (call_map if is_call else put_map)[expiry] = {}
            all_expiries.add(expiry)
            # Handle both Schwab live {strike: [contract]} and mock [contract] formats
            if isinstance(options, dict):
                items = []
                for strike_str, clist in options.items():
                    if isinstance(clist, list): items += clist
                    elif isinstance(clist, dict): items.append(clist)
            else:
                items = options if isinstance(options, list) else []
            for c in items:
                if not isinstance(c, dict): continue
                strike = c.get("strikePrice", 0)
                gamma  = abs(c.get("gamma", 0))
                # For index options (SPX/SPXW), Schwab always returns openInterest=0
                # Use totalVolume as proxy — it reflects real daily activity
                oi = c.get("totalVolume") or c.get("openInterest") or 0
                gex = gamma * oi * 100 * (spot ** 2) / 1e9
                gex = round(gex, 1)
                if is_call:
                    call_map[expiry][strike] = gex
                else:
                    put_map[expiry][strike]  = -gex

    _parse_contracts(chain.get("callExpDateMap", {}), True)
    _parse_contracts(chain.get("putExpDateMap",  {}), False)

    # Sort expiries, take first `days`
    sorted_expiries = sorted(all_expiries)[:days]

    # Build strike range around ATM
    all_strikes = set()
    for d in [call_map, put_map]:
        for exp, strikes in d.items():
            all_strikes.update(strikes.keys())

    if not all_strikes:
        raise HTTPException(status_code=400, detail="No options data found")

    atm_strike = min(all_strikes, key=lambda s: abs(s - spot))
    sorted_all = sorted(all_strikes)
    atm_idx    = sorted_all.index(atm_strike)
    lo = max(0, atm_idx - strike_range)
    hi = min(len(sorted_all), atm_idx + strike_range + 1)
    strikes_to_show = sorted_all[lo:hi]

    # Build heatmap rows
    rows = []
    net_gex_total = 0
    call_wall_strike, call_wall_val = 0, 0
    put_wall_strike,  put_wall_val  = 0, 0

    for strike in reversed(strikes_to_show):
        # Exactly one ATM: the single nearest strike to spot
        is_atm = (strike == atm_strike)
        row = {"strike": strike, "is_atm": is_atm, "cells": {}}
        row_total = 0
        for exp in sorted_expiries:
            call_gex = call_map.get(exp, {}).get(strike, 0)
            put_gex  = put_map.get(exp,  {}).get(strike, 0)
            net      = round(call_gex + put_gex, 1)
            row["cells"][exp] = net
            row_total += net

        row["total"] = round(row_total, 1)
        net_gex_total += row_total

        # Track walls
        if row_total > call_wall_val and strike > spot:
            call_wall_val = row_total; call_wall_strike = strike
        if row_total < put_wall_val and strike < spot:
            put_wall_val = row_total; put_wall_strike = strike

        rows.append(row)

    # Pin strike = highest absolute GEX
    pin_row = max(rows, key=lambda r: abs(r["total"]), default=None)

    # Scale for color intensity
    all_vals = [c for r in rows for c in r["cells"].values() if c != 0]
    max_abs  = max((abs(v) for v in all_vals), default=1)

    return {
        "symbol":       sym,
        "spot":         spot,
        "expiries":     sorted_expiries,
        "strikes":      [r["strike"] for r in rows],
        "rows":         rows,
        "max_abs":      round(max_abs, 1),
        "net_gex":      round(net_gex_total, 1),
        "pin_strike":   pin_row["strike"] if pin_row else None,
        "call_wall":    call_wall_strike,
        "put_wall":     put_wall_strike,
        "atm_strike":   atm_strike,
    }


def _generate_mock_gex_chain(symbol: str, days: int, strike_range: int) -> dict:
    """Multi-expiry mock chain for GEX heatmap."""
    import random, datetime
    prices = {"SPY":590,"QQQ":505,"SPX":5900,"SPXW":5900,"NVDA":138,"AAPL":221,"TSLA":295,"META":632,"MSFT":415,"AMZN":227}
    spot   = prices.get(symbol.upper(), 100) * random.uniform(0.997, 1.003)
    step   = round(spot * 0.002, 0) or 1  # ~0.2% steps

    # Generate business day expiries
    expiries = []
    d = datetime.date.today()
    while len(expiries) < days:
        if d.weekday() < 5:
            expiries.append(str(d))
        d += datetime.timedelta(days=1)

    call_map, put_map = {}, {}
    for exp in expiries:
        call_map[exp] = []
        put_map[exp]  = []
        for i in range(-strike_range, strike_range + 1):
            strike = round(spot + i * step, 1)
            dist   = abs(i) / strike_range
            # OI peaks near ATM, drops off with distance
            base_oi = int(random.uniform(500, 3000) * (1 - dist * 0.7))
            # Gamma peaks near ATM
            gamma   = max(0.001, 0.08 * (1 - dist * 0.85) * random.uniform(0.7, 1.3))

            call_map[exp].append({"strikePrice": strike, "gamma": gamma,
                "openInterest": base_oi + random.randint(0, 2000),
                "putCall": "CALL", "daysToExpiration": 1})
            put_map[exp].append({"strikePrice": strike, "gamma": gamma,
                "openInterest": base_oi + random.randint(0, 2000),
                "putCall": "PUT",  "daysToExpiration": 1})

    return {
        "symbol": symbol.upper(),
        "underlyingPrice": round(spot, 2),
        "callExpDateMap": call_map,
        "putExpDateMap":  put_map,
        "status": "SUCCESS",
    }


# ----------------------------------------------------------------
# FLOW endpoints
# ----------------------------------------------------------------

from core.flow import scan_flow, generate_mock_flow, summarize_flow


FLOW_WATCHLIST = [
    "SPY","QQQ","NVDA","AAPL","TSLA","META","MSFT","AMZN",
    "AMD","COIN","GOOGL","NFLX","GLD","SMH","XLE","SOFI","PLTR"
]


@app.get("/flow")
async def get_flow(
    min_score:    int   = Query(55,    description="Min conviction score"),
    min_notional: float = Query(25000, description="Min dollar notional"),
    min_vol_oi:   float = Query(0.3,   description="Min vol/OI ratio"),
    contract_type:str   = Query("ALL", description="CALL / PUT / ALL"),
    filter_hedges:bool  = Query(True,  description="Filter out hedging prints"),
    exclude:      str   = Query(None,  description="Comma-separated symbols to exclude"),
    symbols:      str   = Query(None,  description="Comma-separated symbols"),
):
    """Scan for unusual options flow across watchlist."""
    tickers = [s.strip().upper() for s in symbols.split(",")] if symbols else FLOW_WATCHLIST

    if USE_LIVE_DATA:
        chains = []
        for sym in tickers:
            try:
                chain = fetch_chain(sym, max_dte=60)
                chains.append({"symbol": sym, "chain": chain, "quote": fetch_quote(sym)})
            except Exception as e:
                pass
        prints = scan_flow(
            chains=chains,
            min_score=min_score,
            min_notional=min_notional,
            min_vol_oi=min_vol_oi,
            contract_type=contract_type,
            filter_hedges=filter_hedges,
        )
    else:
        prints = generate_mock_flow(tickers)
        # Apply filters to mock data
        if contract_type != "ALL":
            prints = [p for p in prints if p["type"] == contract_type]
        prints = [p for p in prints if p["score"] >= min_score and p["notional"] >= min_notional]

    # Apply exclude filter
    if exclude:
        excl_set = {s.strip().upper() for s in exclude.split(",")}
        prints = [p for p in prints if p.get("symbol", "").upper() not in excl_set]

    summary = summarize_flow(prints)

    total_bull = sum(p["notional"] for p in prints if p["direction"] == "BULLISH")
    total_bear = sum(p["notional"] for p in prints if p["direction"] == "BEARISH")
    total      = total_bull + total_bear

    return {
        "timestamp":     datetime.now().isoformat(),
        "mode":          "LIVE" if USE_LIVE_DATA else "MOCK",
        "found":         len(prints),
        "prints":        prints,
        "summary":       summary,
        "market_bias":   "BULLISH" if total_bull > total_bear * 1.3
                         else "BEARISH" if total_bear > total_bull * 1.3
                         else "MIXED",
        "total_bullish_notional": round(total_bull),
        "total_bearish_notional": round(total_bear),
        "filters": {
            "min_score":    min_score,
            "min_notional": min_notional,
            "min_vol_oi":   min_vol_oi,
            "contract_type":contract_type,
            "filter_hedges":filter_hedges,
        }
    }
