# Automating mlbhit daily runs

Two-file setup: a launchd plist that schedules the job, and a shell script
that actually runs it.

## Local automation (Mac stays on, no cloud)

Edit `com.mlbhit.daily.plist` and replace `REPLACE_ME` with your username
(your home folder is `/Users/$(whoami)`). Then:

```bash
cp launchd/com.mlbhit.daily.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.mlbhit.daily.plist

# Verify it's loaded
launchctl list | grep com.mlbhit

# Test once immediately (bypasses the schedule)
launchctl start com.mlbhit.daily
tail -f logs/*.log
```

If the job fails silently, check `logs/launchd.err` and `logs/launchd.out`.
Common first-run failures are (1) wrong path in the plist, (2) venv python
missing, (3) `xcode-select --install` not run (breaks pybaseball).

To change the schedule, edit the plist, then `unload` + `load`. launchd does
not hot-reload.

## Remote automation (laptop closed, anywhere)

Three options, roughly in order of effort:

### Option A — Mac mini + Tailscale (your current box, access from anywhere)
If this mini stays powered on at home, the launchd setup above already gives
you remote access. Add Tailscale (`brew install --cask tailscale`) and you
can SSH in from any device to check logs, trigger runs, or paste prop CSVs.
Zero incremental cost beyond the Mac mini you own.

### Option B — GitHub Actions (free, public repo; cheap, private repo)
Move the repo to GitHub, add a workflow triggered on `schedule:` cron. Secrets
(API keys for odds, etc.) live in repo settings. Outputs (predictions,
recommendations parquet) commit back to the repo or write to S3. Good fit if
you want to collaborate with someone or treat the project as production.
Caveat: needs all data (Statcast cache, model artifacts) to be either
reproducible from scratch each run or stored remotely.

### Option C — Cloud VM (DigitalOcean / Hetzner / Fly, ~$5–10 / mo)
Rent a small Linux box, rsync the project + data to it, set up a crontab
running the same daily_runner.sh (needs bash instead of zsh tweaks). Survives
your laptop dying. Most flexibility. Requires rsync'ing model artifacts and
raw data up to the VM, and a plan for how new Statcast/boxscore data flows
back down if you still work locally.

## The runner script

`scripts/daily_runner.sh` is the single entry point. It:

1. Pulls today's schedule
2. Pulls today's lineups (re-run in afternoon once lineups are posted)
3. Scores with the blended (current + prior season) feature path
4. If a `data/raw/props/YYYY-MM-DD_props.csv` exists, loads props and runs
   the EV recommender
5. Writes a timestamped log per run to `logs/`

Invoke manually:

```bash
./scripts/daily_runner.sh              # scores today
./scripts/daily_runner.sh 2026-06-15   # scores a past date for review
```

## Rolling your own schedule

If the twice-daily schedule doesn't fit your workflow, the plist takes
multiple `StartCalendarInterval` entries. Examples:

- Every 30 minutes on game days (overkill but useful while debugging):
  set `Hour` ranges, omit `Minute` to get every minute of that hour.
- Sunday-only weekly Statcast refresh: add a separate plist
  `com.mlbhit.weekly.plist` that runs `fetch_stats_from_statcast` with
  `--seasons 2023 2024 2025 2026`.

Apple's launchd docs are terse but accurate: `man launchd.plist` on your Mac.
