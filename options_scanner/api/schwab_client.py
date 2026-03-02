"""
schwab_client.py
----------------
Handles OAuth2 authentication and all Schwab Market Data API calls.
Docs: https://developer.schwab.com/products/trader-api--individual-/details/documentation/Retail%20Trader%20API%20Production
"""

import time
import base64
import requests
from datetime import datetime, timedelta
from typing import Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import SCHWAB_APP_KEY, SCHWAB_APP_SECRET, SCHWAB_CALLBACK_URL


# Schwab API base URLs
AUTH_URL     = "https://api.schwabapi.com/v1/oauth/authorize"
TOKEN_URL    = "https://api.schwabapi.com/v1/oauth/token"
MARKET_BASE  = "https://api.schwabapi.com/marketdata/v1"


class SchwabClient:
    """
    Lightweight Schwab API client.
    Handles token refresh automatically.
    """

    TOKEN_FILE = "/app/schwab_tokens.json"  # persists across requests

    def __init__(self):
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.token_expires_at: float = 0
        self._load_tokens()  # load from disk on startup

    def _load_tokens(self):
        """Load tokens from disk if they exist."""
        import json, os
        try:
            if os.path.exists(self.TOKEN_FILE):
                with open(self.TOKEN_FILE, 'r') as f:
                    data = json.load(f)
                self.access_token     = data.get("access_token")
                self.refresh_token    = data.get("refresh_token")
                self.token_expires_at = data.get("token_expires_at", 0)
                print(f"✅ Tokens loaded from {self.TOKEN_FILE}")
        except Exception as e:
            print(f"⚠️  Could not load tokens: {e}")

    # ------------------------------------------------------------------
    # STEP 1: Get the authorization URL — open this in your browser once
    # ------------------------------------------------------------------
    def get_auth_url(self) -> str:
        """
        Returns the URL to open in your browser to authorize the app.
        After authorizing, Schwab redirects to your callback URL with a 'code' param.
        """
        return (
            f"{AUTH_URL}"
            f"?client_id={SCHWAB_APP_KEY}"
            f"&redirect_uri={SCHWAB_CALLBACK_URL}"
            f"&response_type=code"
            f"&scope=readonly"
        )

    # ------------------------------------------------------------------
    # STEP 2: Exchange the auth code for tokens
    # ------------------------------------------------------------------
    def exchange_code_for_tokens(self, auth_code: str) -> dict:
        """
        Call this once with the 'code' from the callback URL.
        Saves access + refresh tokens.
        """
        credentials = base64.b64encode(
            f"{SCHWAB_APP_KEY}:{SCHWAB_APP_SECRET}".encode()
        ).decode()

        response = requests.post(
            TOKEN_URL,
            headers={
                "Authorization": f"Basic {credentials}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={
                "grant_type": "authorization_code",
                "code": auth_code,
                "redirect_uri": SCHWAB_CALLBACK_URL,
            },
        )
        response.raise_for_status()
        tokens = response.json()
        self._save_tokens(tokens)
        return tokens

    # ------------------------------------------------------------------
    # Token refresh (happens automatically)
    # ------------------------------------------------------------------
    def _refresh_access_token(self):
        if not self.refresh_token:
            raise RuntimeError("No refresh token — run exchange_code_for_tokens() first.")

        credentials = base64.b64encode(
            f"{SCHWAB_APP_KEY}:{SCHWAB_APP_SECRET}".encode()
        ).decode()

        response = requests.post(
            TOKEN_URL,
            headers={
                "Authorization": f"Basic {credentials}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={
                "grant_type": "refresh_token",
                "refresh_token": self.refresh_token,
            },
        )
        response.raise_for_status()
        self._save_tokens(response.json())

    def _save_tokens(self, tokens: dict):
        import json
        self.access_token    = tokens["access_token"]
        self.refresh_token   = tokens.get("refresh_token", self.refresh_token)
        expires_in           = tokens.get("expires_in", 1800)
        self.token_expires_at = time.time() + expires_in - 60
        # Persist to disk so tokens survive restarts
        try:
            with open(self.TOKEN_FILE, 'w') as f:
                json.dump({
                    "access_token":     self.access_token,
                    "refresh_token":    self.refresh_token,
                    "token_expires_at": self.token_expires_at,
                }, f)
        except Exception as e:
            print(f"⚠️  Could not save tokens: {e}")

    def _get_headers(self) -> dict:
        if time.time() >= self.token_expires_at:
            self._refresh_access_token()
        return {"Authorization": f"Bearer {self.access_token}"}

    # ------------------------------------------------------------------
    # Market Data: Options Chain
    # ------------------------------------------------------------------
    def get_options_chain(
        self,
        symbol: str,
        contract_type: str = "ALL",   # "CALL", "PUT", or "ALL"
        days_to_exp: int = 45,
        strike_count: int = 20,       # strikes above + below ATM
    ) -> dict:
        """
        Fetches the full options chain for a symbol.
        Returns raw Schwab API response dict.
        """
        from_date = datetime.now().strftime("%Y-%m-%d")
        to_date   = (datetime.now() + timedelta(days=days_to_exp)).strftime("%Y-%m-%d")

        resp = requests.get(
            f"{MARKET_BASE}/chains",
            headers=self._get_headers(),
            params={
                "symbol":       symbol,
                "contractType": contract_type,
                "strikeCount":  strike_count,
                "fromDate":     from_date,
                "toDate":       to_date,
                "optionType":   "S",  # Standard options only (no weeklies filter here)
            },
        )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Market Data: Quote (underlying price)
    # ------------------------------------------------------------------
    def get_quote(self, symbol: str) -> dict:
        """Returns current quote for a stock symbol."""
        resp = requests.get(
            f"{MARKET_BASE}/quotes/{symbol}",
            headers=self._get_headers(),
        )
        resp.raise_for_status()
        return resp.json()
