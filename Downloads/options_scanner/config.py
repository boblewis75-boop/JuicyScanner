# ============================================================
#  JuicyScanner — Configuration
#
#  LOCAL:  Set values directly below
#  CLOUD:  Set these as environment variables in Railway dashboard
#          (never commit real API keys to git)
# ============================================================

import os

# --- Schwab API Credentials ---
SCHWAB_APP_KEY      = os.environ.get("SCHWAB_APP_KEY",     "YOUR_APP_KEY_HERE")
SCHWAB_APP_SECRET   = os.environ.get("SCHWAB_APP_SECRET",  "YOUR_APP_SECRET_HERE")
SCHWAB_CALLBACK_URL = os.environ.get("SCHWAB_CALLBACK_URL","https://juicyscanner-production.up.railway.app/auth/callback")

# --- Mode ---
# Set USE_LIVE_DATA=true in Railway Variables to go live
USE_LIVE_DATA = os.environ.get("USE_LIVE_DATA", "false").lower() == "true"

# --- Scanner defaults ---
DEFAULT_MIN_RR     = float(os.environ.get("DEFAULT_MIN_RR",    "1.5"))
DEFAULT_MIN_TARGET = float(os.environ.get("DEFAULT_MIN_TARGET", "3.0"))
DEFAULT_MAX_DTE    = int(os.environ.get("DEFAULT_MAX_DTE",      "45"))
DEFAULT_MIN_SCORE  = int(os.environ.get("DEFAULT_MIN_SCORE",    "60"))
DEFAULT_MIN_OI     = int(os.environ.get("DEFAULT_MIN_OI",       "50"))
DEFAULT_MAX_SPREAD = float(os.environ.get("DEFAULT_MAX_SPREAD", "0.35"))

# --- Watchlist ---
_wl = os.environ.get("WATCHLIST", "")
WATCHLIST = _wl.split(",") if _wl else [
    # Mega cap tech (highest options volume on the market)
    "AAPL", "MSFT", "NVDA", "META", "GOOGL", "AMZN", "TSLA",

    # Semis — always active
    "AMD",  "MU",   "INTC", "QCOM", "ARM",  "AVGO", "TSM",
    "SMCI", "MRVL", "AMAT", "LRCX", "KLAC", "ON",

    # High-beta / meme / retail favorites
    "COIN", "PLTR", "SOFI", "HOOD", "RBLX", "SNAP", "DKNG",
    "MARA", "RIOT", "CLSK", "CORZ", "HUT",  "CIFR",

    # Mobility / consumer
    "UBER", "LYFT", "ABNB", "DASH", "BKNG",

    # Mag-7 adjacent / AI plays
    "ORCL", "CRM",  "NOW",  "ADBE", "INTU",

    # Cybersecurity
    "CRWD", "PANW", "ZS",   "NET",  "S",    "FTNT",

    # Cloud / SaaS
    "SNOW", "DDOG", "MDB",  "GTLB", "CFLT", "HUBS",

    # Financials
    "GS",   "JPM",  "BAC",  "C",    "MS",   "WFC",
    "V",    "MA",   "AXP",  "PYPL",

    # Biotech / pharma (volatile, great for options)
    "MRNA", "BNTX", "NVAX", "SAVA", "BIIB", "GILD",
    "LLY",  "NVO",

    # Energy
    "XOM",  "CVX",  "OXY",  "SLB",  "HAL",

    # Retail / consumer / media
    "NFLX", "DIS",  "SHOP", "ETSY", "PINS", "ROKU",
    "NKE",  "LULU",

    # Commodities / macro
    "GLD",  "SLV",  "USO",  "GDX",
]

# --- Server ---
HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", "8000"))
