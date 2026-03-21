#!/usr/bin/env python3
"""
schwab_auth.py — One-time Schwab OAuth token generator
=======================================================

Run this ONCE on your Windows machine to get a token.json.
After that, spx_nodes.py and JuicyScanner will auto-refresh it.

Setup:
  1. pip install requests python-dotenv
  2. Create a .env file next to this script:
       SCHWAB_APP_KEY=your_app_key_here
       SCHWAB_APP_SECRET=your_app_secret_here
       SCHWAB_REDIRECT_URI=https://127.0.0.1
  3. Run:  python schwab_auth.py
  4. A browser window opens — log in to Schwab and approve access
  5. You'll be redirected to https://127.0.0.1/?code=...
     Copy the FULL URL from your browser bar and paste it here
  6. token.json is saved next to this script

Windows note: if the browser doesn't open automatically,
copy the printed URL and paste it into Chrome/Edge manually.
"""

import os, sys, json, time, base64, webbrowser
from urllib.parse import urlparse, parse_qs, urlencode
from dotenv import load_dotenv
import requests

load_dotenv()

APP_KEY      = os.getenv("SCHWAB_APP_KEY")
APP_SECRET   = os.getenv("SCHWAB_APP_SECRET")
REDIRECT_URI = os.getenv("SCHWAB_REDIRECT_URI", "https://127.0.0.1")
TOKEN_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "token.json")
API          = "https://api.schwabapi.com"

if not APP_KEY or not APP_SECRET:
    print("ERROR: SCHWAB_APP_KEY and SCHWAB_APP_SECRET must be set in .env")
    sys.exit(1)


def get_auth_url() -> str:
    params = {
        "response_type": "code",
        "client_id":     APP_KEY,
        "redirect_uri":  REDIRECT_URI,
        "scope":         "readonly",
    }
    return f"{API}/v1/oauth/authorize?" + urlencode(params)


def exchange_code(code: str) -> dict:
    creds = base64.b64encode(f"{APP_KEY}:{APP_SECRET}".encode()).decode()
    r = requests.post(
        f"{API}/v1/oauth/token",
        headers={
            "Authorization":  f"Basic {creds}",
            "Content-Type":   "application/x-www-form-urlencoded",
        },
        data={
            "grant_type":   "authorization_code",
            "code":          code,
            "redirect_uri":  REDIRECT_URI,
        },
    )
    if not r.ok:
        print(f"Token exchange failed: {r.status_code} {r.text}")
        sys.exit(1)
    return r.json()


def main():
    print("=" * 60)
    print("  Schwab OAuth — JuicyScanner Token Setup")
    print("=" * 60)

    auth_url = get_auth_url()
    print(f"\nStep 1: Opening Schwab login in your browser...")
    print(f"\n  {auth_url}\n")
    print("(If it doesn't open, copy the URL above into Chrome/Edge)")

    try:
        webbrowser.open(auth_url)
    except Exception:
        pass

    print("\nStep 2: Log in and approve access.")
    print("        After approval, your browser will redirect to a")
    print("        page that says 'This site can't be reached' — that's OK.")
    print()
    print("Step 3: Copy the FULL URL from your browser address bar")
    print("        It will look like:  https://127.0.0.1/?code=C0.AAAA...&session=...")
    print()

    callback_url = input("Paste the full redirect URL here: ").strip()

    # Parse the code out of the URL
    try:
        parsed = urlparse(callback_url)
        params = parse_qs(parsed.query)
        code   = params.get("code", [None])[0]
        if not code:
            raise ValueError("No 'code' parameter found in URL")
    except Exception as e:
        print(f"\nERROR parsing URL: {e}")
        print("Make sure you pasted the complete URL including 'https://'")
        sys.exit(1)

    print(f"\nExchanging authorization code for tokens...")
    token = exchange_code(code)
    token["creation_timestamp"] = int(time.time())

    with open(TOKEN_PATH, "w") as f:
        json.dump(token, f, indent=2)

    print(f"\n✅ SUCCESS! token.json saved to:")
    print(f"   {TOKEN_PATH}")
    print(f"\n   Access token expires in: {token.get('expires_in', '?')} seconds")
    print(f"   Refresh token expires in: {token.get('refresh_token_expires_in', '?')} seconds")
    print(f"\nYou can now run:  python spx_nodes.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
