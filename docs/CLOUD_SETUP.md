# Cloud setup — daily picks on autopilot

This is a one-time, ~10-minute setup. After this, picks land in your inbox at
7am ET every day, and you can re-run from your phone whenever lineups update.

If you get stuck on any step, stop and tell me which one. We'll fix it together.

---

## What you'll end up with

- An automatic run every morning at **7:00 AM ET** that produces today's
  Filter E picks.
- An **email** with a clean summary table and the full CSV attached.
- A **bookmarkable webpage** (your private dashboard) showing the latest picks,
  optimized for your phone. You can save it to your home screen and it'll
  look like an app.
- A **"Run workflow"** button you can tap from the GitHub mobile app to
  refresh picks mid-afternoon when lineups firm up.
- Zero changes to your Mac. It can be off, asleep, anywhere — picks still come.

## What stays private

- The repository is **private** — only you can see the code, picks, model.
- Your `ODDS_API_KEY` lives in **GitHub Secrets** — encrypted, never visible
  in code or logs.
- The dashboard URL (`https://<your-username>.github.io/mlb-hit/`) is
  **technically public**, but unguessable. If that bothers you, we can put it
  behind GitHub authentication later — see "Optional hardening" below.

---

## Step 1 — Create a GitHub account (skip if you already have one)

1. Go to **https://github.com/signup**
2. Use your real email so the daily digest can come back to it.
3. **Turn on 2-factor authentication immediately.** Settings → Password and
   authentication → "Two-factor authentication." Use the authenticator app
   option (Authy, 1Password, or Google Authenticator).
4. Install the **GitHub mobile app** from the App Store. Sign in.
5. Generate a personal access token only if you're going to push from the
   command line (we will be — see Step 4).

## Step 2 — Create a Mailgun account for sending the email digest

Free tier easily covers one email a day.

1. Go to **https://signup.mailgun.com/new/signup** and create an account.
2. They'll ask you to add a sending domain. Two options:
   - **Easiest:** use the sandbox domain they give you for free
     (`sandboxXXXXX.mailgun.org`). You'll have to add your inbox as an
     "authorized recipient" — Mailgun emails you a confirmation link.
   - **Cleaner:** add your own domain. Skip this until you've confirmed the
     sandbox flow works.
3. Go to **Sending → Domain settings → API keys**. Copy the value labeled
   **"Private API key"** (starts with `key-` or is a 32-char hex string).
4. Note your sending domain (e.g. `sandbox123abc.mailgun.org`). You'll need
   both for Step 6.

## Step 3 — Create the private GitHub repo

You'll do this once on github.com:

1. Go to **https://github.com/new**
2. Repository name: `mlb-hit` (or whatever you want, no spaces).
3. Set it to **Private**.
4. Don't initialize with a README — we'll push the existing code.
5. Click **Create repository**.
6. Copy the URL it shows you (looks like `https://github.com/yourname/mlb-hit.git`).

## Step 4 — Push the code to the new repo

On your Mac, in Terminal:

```bash
cd "$HOME/Documents/Claude/Projects/mlb hit"

# If this folder isn't already a git repo:
git init -b main
git add .
git commit -m "Initial production import"

# Point at the new GitHub repo (paste the URL from Step 3):
git remote add origin https://github.com/YOURNAME/mlb-hit.git

# Create the production branch — this is what the cloud runner uses:
git branch production
git tag prod-xgb_v3

# Push everything:
git push -u origin main production
git push origin prod-xgb_v3
```

When prompted for a password, use a **personal access token**, not your
GitHub password. Create one at **https://github.com/settings/tokens** →
"Generate new token (classic)" → check `repo` scope. Copy the token; you
won't see it again.

## Step 5 — Make the runner use the `production` branch

In the GitHub UI, on your new repo:

1. **Settings → Branches** → set the default branch to `production`.
2. **Settings → Branches → branch protection rules** → add rule for
   `production` → check "Require pull request before merging." This stops
   accidental direct pushes to production once you start experimenting on
   `main`.

## Step 6 — Add the secrets

On your repo: **Settings → Secrets and variables → Actions → New repository secret**.

Add **five** secrets, one at a time. Names must match exactly:

| Name              | Value                                                  |
|-------------------|--------------------------------------------------------|
| `ODDS_API_KEY`    | Your the-odds-api key                                  |
| `MAILGUN_API_KEY` | The "Private API key" from Mailgun (Step 2)            |
| `MAILGUN_DOMAIN`  | Your Mailgun sending domain (e.g. `sandbox....org`)    |
| `DIGEST_TO`       | The email address you want the digest sent to          |
| `DIGEST_FROM`     | A "From" address on your Mailgun domain (e.g. `picks@sandbox....org`) |

## Step 7 — Enable GitHub Pages for the dashboard

1. **Settings → Pages**.
2. **Source:** Deploy from a branch.
3. **Branch:** `production` / folder: `/docs`.
4. Click **Save**.
5. Wait ~1 minute. The page tells you the URL —
   `https://yourname.github.io/mlb-hit/`. Bookmark it on your phone.
   - On iPhone: open the URL in Safari → tap the share icon → "Add to Home
     Screen." It'll look and feel like an app.

The dashboard will say "no manifest found" until the first workflow run
finishes — that's expected.

## Step 8 — Run the workflow once to confirm everything works

1. On your repo: **Actions → Daily Picks → Run workflow** (top right).
2. Leave the date input blank. Click the green **Run workflow** button.
3. Watch the run. It should finish in 2–5 minutes.
4. You should get an email with today's picks.
5. Refresh the dashboard URL — picks should appear.

If the run fails, click into the failed step and copy the red error message.
That tells us exactly what to fix.

---

## Daily life after setup

- **Morning:** email arrives at ~7:05am ET with today's picks. Done.
- **Mid-afternoon (if you want a refresh after late lineups post):** open the
  GitHub mobile app → your repo → Actions tab → **Daily Picks** → tap the
  three-dot menu → **Run workflow**. Email arrives in 2-3 minutes with
  updated picks.
- **Anytime:** open your dashboard bookmark to scan today's bets.

## Optional hardening (do later, not now)

- **Custom email domain** instead of the sandbox.
- **Password-protected dashboard:** instead of GitHub Pages, deploy the
  dashboard to Cloudflare Pages with Cloudflare Access in front (free, takes
  ~10 min). Reach out when you want this.
- **Slack/Discord webhook** instead of (or in addition to) email — easy add.

## When you experiment without breaking production

- Create a new branch off `main` (`git checkout -b experiment-2026-05-foo`).
- All your changes go there. The `production` branch is untouched.
- If you want to promote a new model:
  1. Backtest it locally to your satisfaction.
  2. Merge into `main`, then merge `main` into `production`.
  3. Tag the commit (`git tag prod-xgb_v4 && git push origin prod-xgb_v4`).
  4. Next cron run uses the new model.
- To **roll back instantly**: `git reset --hard prod-xgb_v3 && git push -f origin production`.
  Even faster: in the GitHub UI, you can "Revert" any commit on `production`
  with one click. The next cron run is back on the old version.

---

## How much this costs

- **GitHub Actions:** free up to 2,000 min/month on private repos. We use ~3
  min/day = ~90 min/month.
- **GitHub Pages:** free.
- **Mailgun sandbox:** free for 100 emails/day. We send 1–2/day.
- **the-odds-api:** unchanged from what you're paying today.

**Total monthly added cost: $0.**
