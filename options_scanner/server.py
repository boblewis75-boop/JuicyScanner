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
from core.brain import (
    get_state as brain_state,
    start_tracking as brain_track,
    close_play as brain_close,
    update_entry as brain_update_entry,
    reset_weights as brain_reset,
    auto_log_scan as brain_auto_log,
)
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

    # Fetch flow once for all symbols (shared context)
    try:
        from core.flow import scan_flow, generate_mock_flow
        if USE_LIVE_DATA:
            flow_prints = scan_flow(get_schwab_client(), tickers[:10])
        else:
            flow_prints = generate_mock_flow(tickers[:10])
        shared_flow = {"prints": flow_prints}
    except Exception:
        shared_flow = None

    def _scan_one(symbol):
        try:
            chain = fetch_chain(symbol, max_dte)
            quote = fetch_quote(symbol)
            bars  = generate_mock_bars(symbol)
            # Fetch GEX for this symbol
            gex_d = None
            try:
                from core.scanner import score_gex_confluence
                if USE_LIVE_DATA:
                    gex_chain = get_schwab_client().get_options_chain(
                        symbol, days_to_exp=max_dte*5, contract_type="ALL")
                    spot = gex_chain.get("underlyingPrice") or quote.get("lastPrice", 0)
                    # Build minimal gex summary from chain
                    gex_d = _build_gex_summary(gex_chain, spot)
            except Exception:
                gex_d = None
            plays = scan_options(
                chain, quote, bars=bars,
                min_rr=min_rr, min_target=min_target, max_dte=max_dte,
                min_score=min_score, min_oi=min_oi, contract_type=contract_type,
                gex_data=gex_d, flow_data=shared_flow, auto_save=True,
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
        # Fetch GEX + flow for single-symbol scan
        gex_d = None; flow_d = None
        try:
            if USE_LIVE_DATA:
                gex_chain = get_schwab_client().get_options_chain(
                    symbol, days_to_exp=max_dte*5, contract_type="ALL")
                spot  = gex_chain.get("underlyingPrice") or quote.get("lastPrice", 0)
                gex_d = _build_gex_summary(gex_chain, spot)
            from core.flow import scan_flow, generate_mock_flow
            flow_prints = scan_flow(get_schwab_client(), [symbol]) if USE_LIVE_DATA else generate_mock_flow([symbol])
            flow_d = {"prints": flow_prints}
        except Exception:
            pass
        plays = scan_options(
            chain, quote, bars=bars,
            min_rr=min_rr, min_target=min_target, max_dte=max_dte,
            min_score=min_score, contract_type=contract_type,
            gex_data=gex_d, flow_data=flow_d, auto_save=True,
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

# ─────────────────────────────────────────────────────────────────
# GEX summary helper (used by scan endpoints)
# ─────────────────────────────────────────────────────────────────

def _build_gex_summary(chain: dict, spot: float) -> dict:
    """Build a minimal GEX summary dict from a raw Schwab chain."""
    import math as _math
    call_map = {}; put_map = {}

    def _parse(exp_map, is_call):
        for exp_key, options in exp_map.items():
            if not isinstance(options, dict): continue
            for strike_str, clist in options.items():
                contracts = clist if isinstance(clist, list) else [clist]
                for c in contracts:
                    if not isinstance(c, dict): continue
                    strike = c.get("strikePrice", 0)
                    gamma  = abs(c.get("gamma", 0) or 0)
                    oi     = c.get("totalVolume") or c.get("openInterest") or 0
                    gex    = gamma * oi * 100 * (spot ** 2) / 1e9
                    m = call_map if is_call else put_map
                    m[strike] = m.get(strike, 0) + (gex if is_call else -gex)

    _parse(chain.get("callExpDateMap", {}), True)
    _parse(chain.get("putExpDateMap",  {}), False)

    all_strikes = sorted(set(list(call_map.keys()) + list(put_map.keys())))
    if not all_strikes:
        return None

    rows = []
    net_total = 0; pin_strike = None; pin_val = 0
    call_wall = None; call_wall_val = 0
    put_wall  = None; put_wall_val  = 0

    for strike in all_strikes:
        total = round((call_map.get(strike, 0) + put_map.get(strike, 0)), 2)
        rows.append({"strike": strike, "total": total})
        net_total += total
        if abs(total) > abs(pin_val):
            pin_val = total; pin_strike = strike
        if total > call_wall_val and strike > spot:
            call_wall_val = total; call_wall = strike
        if total < put_wall_val and strike < spot:
            put_wall_val = total; put_wall = strike

    return {
        "spot": spot, "rows": rows, "net_gex": round(net_total, 1),
        "pin_strike": pin_strike, "call_wall": call_wall, "put_wall": put_wall,
    }


# ─────────────────────────────────────────────────────────────────
# Play History API Routes
# ─────────────────────────────────────────────────────────────────

@app.get("/history")
def get_play_history():
    return brain_state()


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

# ================================================================
# AI Brain endpoints
# ================================================================

@app.get("/brain")
def get_brain():
    return brain_state()

@app.post("/brain/track/{play_id}")
def brain_track_endpoint(play_id: str, entry_premium: float = 0.0, notes: str = ""):
    result = brain_track(play_id, entry_premium=entry_premium or None, notes=notes)
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result.get("error"))
    return result

class BrainCloseRequest(BaseModel):
    exit_premium: float
    result: str
    notes: str = ""

@app.post("/brain/close/{play_id}")
def brain_close_endpoint(play_id: str, req: BrainCloseRequest):
    if req.result not in ("WIN","PARTIAL","LOSS","EXPIRED"):
        raise HTTPException(status_code=400, detail="result must be WIN/PARTIAL/LOSS/EXPIRED")
    r = brain_close(play_id, req.exit_premium, req.result, req.notes)
    if not r["success"]:
        raise HTTPException(status_code=404, detail=r.get("error"))
    return r

class BrainUpdateRequest(BaseModel):
    entry_premium: float = None
    notes: str = None

@app.patch("/brain/plays/{play_id}")
def brain_update_endpoint(play_id: str, req: BrainUpdateRequest):
    r = brain_update_entry(play_id, req.entry_premium, req.notes)
    if not r["success"]:
        raise HTTPException(status_code=404, detail=r.get("error"))
    return r

@app.post("/brain/reset")
def brain_reset_endpoint():
    return brain_reset()

# backwards compat
@app.get("/learning")
def get_learning_compat():
    return brain_state()

@app.post("/learning/reset")
def reset_learning_compat():
    return brain_reset()


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
# Market Sentiment / Fear-Greed endpoint
# ----------------------------------------------------------------

@app.get("/sentiment")
def get_sentiment():
    """
    Returns VIX-based fear/greed score + SPY data.
    Score 0 = extreme fear, 100 = extreme greed.
    """
    try:
        # VIX — Schwab uses $VIX
        vix_sym = "$VIX" if USE_LIVE_DATA else "VIX"
        spy_sym = "SPY"
        
        vix_q = fetch_quote(vix_sym)
        spy_q = fetch_quote(spy_sym)
        
        vix = (vix_q.get("lastPrice") or vix_q.get("mark") or 
               vix_q.get("closePrice") or 18.0)
        spy = (spy_q.get("lastPrice") or spy_q.get("mark") or 
               spy_q.get("closePrice") or 590.0)
        spy_chg = (spy_q.get("netPercentChangeInDouble") or 
                   spy_q.get("regularMarketPercentChange") or 0.0)

        # VIX → score (inverted: high VIX = fear = low score)
        if   vix < 12:  score = 95
        elif vix < 15:  score = 80
        elif vix < 18:  score = 65
        elif vix < 22:  score = 50
        elif vix < 26:  score = 35
        elif vix < 30:  score = 20
        elif vix < 40:  score = 10
        else:           score = 3

        # SPY momentum nudge ±10
        if   spy_chg >  1.5: score = min(100, score + 10)
        elif spy_chg >  0.5: score = min(100, score + 5)
        elif spy_chg < -1.5: score = max(0,   score - 10)
        elif spy_chg < -0.5: score = max(0,   score - 5)

        if   score >= 75: mood = "EXTREME GREED"
        elif score >= 55: mood = "GREED"
        elif score >= 45: mood = "NEUTRAL"
        elif score >= 25: mood = "FEAR"
        else:             mood = "EXTREME FEAR"

        return {
            "score":   score,
            "mood":    mood,
            "vix":     round(float(vix), 2),
            "spy":     round(float(spy), 2),
            "spy_chg": round(float(spy_chg), 2),
            "mode":    "LIVE" if USE_LIVE_DATA else "MOCK",
        }
    except Exception as e:
        # Fallback neutral reading
        return {"score": 50, "mood": "NEUTRAL", "vix": None, "spy": None, "spy_chg": 0, "error": str(e)}

# ----------------------------------------------------------------
# GEX (Gamma Exposure) — superior node-detection engine
# Ported from spx_nodes.py: sign-correct GEX, flip zone, regime labels
# ----------------------------------------------------------------

import math as _math

# ── GEX helpers ────────────────────────────────────────────────

def _compute_gex_map(chain: dict, spot: float) -> dict:
    """
    GEX = sign × |gamma| × OI × 100 × spot
    sign = +1 for calls, -1 for puts
    Returns { strike: { exp_date: gex_value } }

    For index options (SPX/SPXW), Schwab returns openInterest=0 on chain endpoint.
    Fallback order: openInterest → totalVolume → skip.
    Contracts with gamma=0 are skipped (Greeks not populated).
    """
    gex = {}
    total_contracts = 0
    skipped_no_oi   = 0
    skipped_no_gamma = 0

    for side, exp_map in [("CALL", chain.get("callExpDateMap", {})),
                           ("PUT",  chain.get("putExpDateMap",  {}))]:
        for exp_key, options in exp_map.items():
            exp_date = exp_key.split(":")[0] if ":" in exp_key else exp_key
            # Schwab live: {strike_str: [contract, ...]}  |  mock: [contract, ...]
            if isinstance(options, dict):
                items = []
                for sk, clist in options.items():
                    if isinstance(clist, list): items += clist
                    elif isinstance(clist, dict): items.append(clist)
            else:
                items = options if isinstance(options, list) else []

            for c in items:
                if not isinstance(c, dict): continue
                total_contracts += 1
                strike = round(float(c.get("strikePrice", 0)), 2)
                gamma  = c.get("gamma", 0) or 0
                if gamma == 0:
                    skipped_no_gamma += 1
                    continue
                oi = c.get("openInterest", 0) or 0
                if oi == 0:
                    oi = c.get("totalVolume", 0) or 0
                if oi == 0:
                    skipped_no_oi += 1
                    continue
                sign = 1 if side == "CALL" else -1
                val  = sign * abs(gamma) * oi * 100 * spot
                gex.setdefault(strike, {})
                gex[strike][exp_date] = gex[strike].get(exp_date, 0) + val

    # Log diagnostics (visible in Railway logs)
    print(f"[GEX] contracts={total_contracts} skipped_no_gamma={skipped_no_gamma} skipped_no_oi={skipped_no_oi} strikes_with_data={len(gex)}")
    return gex


def _find_flip_zone(net: dict, strikes: list, spot: float) -> float:
    """Strike where net GEX crosses zero — closest to spot."""
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


def _detect_nodes(net: dict, strikes: list, spot: float):
    """
    Regime-aware GEX node labeler.
    POS γ regime: WALL (call wall above), FLOOR (put wall below),
                  GATE (neg node above), TRAP (neg node below)
    NEG γ regime: GATE (neg above), ACCEL (neg below), MGNT (pos magnets)
    Special:      RUG + BREACH (compressed walls with no buffer)
    """
    total_net = sum(net.values())
    mx        = max((abs(v) for v in net.values()), default=1) or 1
    is_neg    = total_net < 0

    above = [s for s in strikes if s > spot]
    below = [s for s in strikes if s < spot]

    def sp(zone): return next(iter(sorted([s for s in zone if net.get(s,0) > 0], key=lambda s: -net[s])), None)
    def sn(zone): return next(iter(sorted([s for s in zone if net.get(s,0) < 0], key=lambda s:  net[s])), None)

    spa = sp(above); spb = sp(below)
    sna = sn(above); snb = sn(below)
    sig = lambda s: s is not None and abs(net.get(s, 0)) / mx > 0.20

    nodes = {}
    if is_neg:
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
    if spa and snb and net.get(spa,0)/mx > 0.45 and abs(net.get(snb,0))/mx > 0.35:
        buffer = [s for s in strikes if snb < s < spa and net.get(s,0) > mx*0.15]
        if not buffer:
            nodes[spa] = "RUG"
            nodes[snb] = "BREACH"

    return nodes, total_net


NODE_COLORS = {
    "WALL":   "#f59e0b",   # amber
    "FLOOR":  "#22c55e",   # green
    "GATE":   "#ef4444",   # red
    "TRAP":   "#f97316",   # orange
    "ACCEL":  "#dc2626",   # dark red
    "MGNT":   "#a855f7",   # purple
    "RUG":    "#ec4899",   # pink
    "BREACH": "#be185d",   # deep pink
}

@app.get("/gex/{symbol}")
def get_gex(
    symbol:       str,
    days:         int = Query(5,  description="Number of expiry dates to include"),
    strike_range: int = Query(60, description="Number of strikes above/below ATM"),
):
    """
    GEX heatmap — sign-correct formula: GEX = sign x |gamma| x OI x 100 x spot
    Includes flip zone, regime detection, and labeled nodes.
    """
    sym = symbol.upper()
    # Schwab requires $SPX / $SPXW for index options
    api_sym = sym
    if sym in ("SPXW", "SPX"):
        api_sym = "$" + sym
    try:
        if USE_LIVE_DATA:
            client = get_schwab_client()
            chain  = client.get_options_chain(api_sym, days_to_exp=days * 10, contract_type="ALL")
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

        # Compute GEX map
        gex_map = _compute_gex_map(chain, spot)
        if not gex_map:
            raise HTTPException(status_code=400, detail="No options data found — gamma may be zero for all contracts")

        # Build strike list within 6% of spot
        all_strikes_raw = sorted(gex_map.keys())
        strikes = [s for s in all_strikes_raw if abs(s - spot) <= spot * 0.06]
        if not strikes:
            strikes = all_strikes_raw

        # Use up to `days` nearest expirations
        all_exps = sorted(set(e for sd in gex_map.values() for e in sd.keys()))
        exps     = all_exps[:days]

        # Net GEX per strike across selected expiries
        net = {}
        for s in strikes:
            net[s] = sum(gex_map.get(s, {}).get(e, 0) for e in exps)

        # Flip zone
        flip = _find_flip_zone(net, strikes, spot)

        # Node detection
        nodes, total_net = _detect_nodes(net, strikes, spot)
        regime     = "NEG_GAMMA" if total_net < 0 else "POS_GAMMA"
        regime_lbl = "NEG gamma — moves AMPLIFY" if total_net < 0 else "POS gamma — moves DAMPEN"

        # Build heatmap rows
        atm_strike = min(strikes, key=lambda s: abs(s - spot))
        mx         = max((abs(v) for v in net.values()), default=1) or 1

        rows = []
        for strike in sorted(strikes, reverse=True):
            cells = {}
            for exp in exps:
                raw = gex_map.get(strike, {}).get(exp, 0)
                cells[exp] = round(raw / 1e6, 3)   # $M for display
            rows.append({
                "strike":     strike,
                "is_atm":     strike == atm_strike,
                "cells":      cells,
                "total":      round(net[strike] / 1e6, 3),
                "intensity":  round(net[strike] / mx, 3),
                "node":       nodes.get(strike),
                "node_color": NODE_COLORS.get(nodes.get(strike), ""),
            })

        # Walls + pin
        call_wall  = next((s for s, l in nodes.items() if l in ("WALL", "RUG")    and s > spot), None)
        put_wall   = next((s for s, l in nodes.items() if l in ("FLOOR", "BREACH") and s < spot), None)
        pin_strike = max(strikes, key=lambda s: abs(net[s]), default=None)

        node_list = sorted(
            [{"strike": s, "label": l, "gex_M": round(net[s] / 1e6, 2),
              "color": NODE_COLORS.get(l, ""), "above_spot": s > spot}
             for s, l in nodes.items()],
            key=lambda x: x["strike"], reverse=True,
        )

        return {
            "symbol":       sym,
            "spot":         spot,
            "expiries":     exps,
            "strikes":      [r["strike"] for r in rows],
            "rows":         rows,
            "max_abs":      round(mx / 1e6, 2),
            "net_gex":      round(total_net / 1e6, 2),
            "net_gex_M":    round(total_net / 1e6, 2),
            "regime":       regime,
            "regime_label": regime_lbl,
            "flip_zone":    flip,
            "flip_above":   flip > spot,
            "pin_strike":   pin_strike,
            "call_wall":    call_wall,
            "put_wall":     put_wall,
            "atm_strike":   atm_strike,
            "nodes":        node_list,
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"[GEX ERROR] {sym}: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"GEX failed for {sym}: {str(e)}")



def _generate_mock_gex_chain(symbol: str, days: int, strike_range: int) -> dict:
    """
    Multi-expiry mock chain for GEX heatmap.
    Generates realistic gamma + OI profiles so the node detector works in mock mode.
    Deliberately places strong nodes at round-number strikes to mimic real SPX structure.
    """
    import random, datetime
    prices = {
        "SPY":590,"QQQ":505,"SPX":5900,"SPXW":5900,
        "NVDA":138,"AAPL":221,"TSLA":295,"META":632,"MSFT":415,"AMZN":227,
        "AMD":112,"COIN":255,"GLD":231,"GOOGL":192,"NFLX":1015,
    }
    spot   = prices.get(symbol.upper(), 200) * random.uniform(0.997, 1.003)
    # SPX-style: $5 steps; equity: ~0.5% steps
    step   = 5 if spot > 1000 else round(spot * 0.005, 0) or 1

    # Round spot to nearest step
    spot = round(round(spot / step) * step, 2)

    # Business day expiries
    expiries = []
    d = datetime.date.today()
    while len(expiries) < days:
        if d.weekday() < 5:
            expiries.append(str(d))
        d += datetime.timedelta(days=1)

    # Choose random "wall" strikes for realism
    wall_above = round(spot + step * random.randint(4, 10), 0)
    wall_below = round(spot - step * random.randint(4, 10), 0)

    call_map, put_map = {}, {}
    for exp in expiries:
        call_contracts = []
        put_contracts  = []
        for i in range(-strike_range, strike_range + 1):
            strike = round(spot + i * step, 1)
            dist   = abs(i) / max(strike_range, 1)

            # Base gamma: bell curve around ATM
            base_gamma = max(0.0001, 0.06 * (1 - dist * 0.9) * random.uniform(0.8, 1.2))

            # Base OI: higher near ATM, spikes at walls
            base_oi = int(random.uniform(200, 2000) * (1 - dist * 0.65))
            if abs(strike - wall_above) < step * 1.5:
                base_oi = int(base_oi * random.uniform(3.0, 6.0))  # call wall spike
            if abs(strike - wall_below) < step * 1.5:
                base_oi = int(base_oi * random.uniform(2.5, 5.0))  # put wall spike

            call_contracts.append({
                "strikePrice": strike, "gamma": base_gamma,
                "openInterest": base_oi + random.randint(0, 500),
                "totalVolume":  random.randint(10, 300),
                "putCall": "CALL", "daysToExpiration": (expiries.index(exp)+1),
            })
            put_contracts.append({
                "strikePrice": strike, "gamma": base_gamma,
                "openInterest": int(base_oi * random.uniform(0.8, 1.4)) + random.randint(0, 300),
                "totalVolume":  random.randint(10, 300),
                "putCall": "PUT",  "daysToExpiration": (expiries.index(exp)+1),
            })

        call_map[exp] = call_contracts
        put_map[exp]  = put_contracts

    return {
        "symbol":           symbol.upper(),
        "underlyingPrice":  round(spot, 2),
        "callExpDateMap":   call_map,
        "putExpDateMap":    put_map,
        "status":           "SUCCESS",
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
