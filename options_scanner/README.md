# OptionsEdge AI — Python Backend

AI-powered options scanner using Schwab Market Data API.  
Works in **mock mode right now** — no API key needed until you're ready.

---

## Quick Start (no API key needed)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the server (runs in mock mode by default)
python server.py

# 3. Open the interactive API docs
open http://localhost:8000/docs

# 4. Run a scan
curl "http://localhost:8000/scan"
```

---

## Project Structure

```
options_scanner/
├── config.py              ← ⭐ Put your API key here when ready
├── server.py              ← FastAPI server (start here)
├── requirements.txt
├── api/
│   └── schwab_client.py   ← Schwab OAuth2 + API calls
├── core/
│   └── scanner.py         ← Scoring engine, R:R calc, filtering
└── data/
    └── mock_data.py        ← Realistic fake data for development
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/scan` | Scan full watchlist |
| GET | `/scan/{symbol}` | Scan single ticker (e.g. `/scan/NVDA`) |
| GET | `/quote/{symbol}` | Get current price |
| GET | `/auth/url` | Get Schwab OAuth URL |
| POST | `/auth/token` | Exchange auth code for tokens |
| GET | `/health` | Server status |

### Scan Parameters

All parameters are optional (defaults in `config.py`):

| Param | Default | Description |
|-------|---------|-------------|
| `min_rr` | 2.0 | Minimum reward:risk ratio |
| `min_target` | 15.0 | Min % move needed in underlying |
| `max_dte` | 45 | Max days to expiration |
| `min_score` | 60 | Min AI score (0-100) |
| `contract_type` | ALL | CALL, PUT, or ALL |
| `symbols` | watchlist | Comma-separated custom list |

**Example:**
```bash
curl "http://localhost:8000/scan?min_rr=3&max_dte=30&contract_type=CALL"
```

---

## Connecting to Schwab (when your key is ready)

### Step 1 — Add credentials to `config.py`
```python
SCHWAB_APP_KEY    = "your_key_here"
SCHWAB_APP_SECRET = "your_secret_here"
USE_LIVE_DATA     = True   # ← flip this
```

### Step 2 — Authorize once
```bash
# Get the auth URL
curl http://localhost:8000/auth/url

# Open that URL in your browser, log in, authorize the app
# Copy the 'code' from the redirect URL, then:
curl -X POST http://localhost:8000/auth/token \
     -H "Content-Type: application/json" \
     -d '{"code": "PASTE_CODE_HERE"}'
```

### Step 3 — You're live!
The server now uses real Schwab data. All endpoints work identically.

---

## How the AI Scoring Works

Each option is scored 0-100 across 6 factors:

| Factor | Weight | Sweet Spot |
|--------|--------|------------|
| R:R Ratio | 25 pts | ≥ 3:1 |
| Target move size | 20 pts | < 18% (higher probability) |
| IV Rank | 20 pts | 25-55 (cheap but not too cheap) |
| Delta | 15 pts | 0.30-0.50 (directional leverage) |
| Liquidity (OI) | 10 pts | Higher is better |
| DTE | 10 pts | 21-35 days (theta sweet spot) |

---

## Connecting to the Frontend

The HTML frontend (`options-scanner.html`) can call this backend directly.
Replace the mock `mockData` array in the HTML with:

```javascript
async function runScan() {
  const resp = await fetch(`http://localhost:8000/scan?min_rr=${minRR}&min_target=${minTarget}&max_dte=${maxDTE}`);
  const data = await resp.json();
  renderTable(data.plays);
}
```

---

*Built with FastAPI + Schwab Market Data API*
