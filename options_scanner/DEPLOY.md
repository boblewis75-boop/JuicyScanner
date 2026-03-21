# 🚀 JuicyScanner — Deploy to Railway (Free)
# Once deployed, open your app on ANY device, ANY browser, NO laptop needed.

## What you need
- A free GitHub account → github.com
- A free Railway account → railway.app (sign in with GitHub)

---

## Step 1 — Put your code on GitHub

1. Go to github.com → click the **+** → **New repository**
2. Name it `juicyscanner` → set to **Private** → click **Create repository**
3. On your computer, open Terminal (Mac) or PowerShell (Windows)
4. Run these commands one at a time:

```bash
cd options_scanner
git init
git add .
git commit -m "JuicyScanner initial deploy"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/juicyscanner.git
git push -u origin main
```
(replace YOUR_USERNAME with your GitHub username)

---

## Step 2 — Deploy on Railway

1. Go to **railway.app** → click **Start a New Project**
2. Choose **Deploy from GitHub repo**
3. Select your `juicyscanner` repo
4. Railway detects Python automatically and starts building

**That's it — Railway builds and deploys in ~2 minutes.**

---

## Step 3 — Get your public URL

1. In Railway dashboard → click your project → click **Settings** tab
2. Under **Networking** → click **Generate Domain**
3. You'll get a URL like: `https://juicyscanner-production.up.railway.app`

**Open that URL on your phone — JuicyScanner is live! 🎉**

---

## Step 4 — Add to your phone home screen

**iPhone (Safari):**
1. Open your Railway URL in Safari
2. Tap the Share button (box with arrow)
3. Tap **"Add to Home Screen"**
4. Name it "JuicyScanner" → tap Add

**Android (Chrome):**
1. Open your Railway URL in Chrome
2. Tap the 3-dot menu
3. Tap **"Add to Home Screen"**

---

## Step 5 — Set environment variables (when you have your Schwab key)

In Railway dashboard → your project → **Variables** tab → click **New Variable**:

| Variable | Value |
|----------|-------|
| `SCHWAB_APP_KEY` | your key from developer.schwab.com |
| `SCHWAB_APP_SECRET` | your secret |
| `USE_LIVE_DATA` | `true` |

Railway automatically restarts the server. No redeployment needed.

---

## Updating the app later

Whenever you change code:
```bash
git add .
git commit -m "describe your change"
git push
```
Railway auto-deploys the new version in ~60 seconds.

---

## Free tier limits

Railway free tier gives you $5/month of compute credit.
JuicyScanner uses ~$1-2/month in idle mode.
More than enough for personal use.

If you go over: upgrade to the $5/month Hobby plan.

---

## Troubleshooting

**App shows "Backend Offline"**
→ Check Railway dashboard → your project → **Deployments** tab for errors

**"Module not found" error**
→ Make sure requirements.txt is in the repo root (the options_scanner folder)

**Deployment stuck at "Building"**
→ Check that runtime.txt says `python-3.11.0`

**Need help?** Railway has great docs at docs.railway.app
