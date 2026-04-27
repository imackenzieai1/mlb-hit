#!/usr/bin/env python
"""Build a manifest of available recommendation CSVs so the GitHub Pages
dashboard can list them and load the latest one without hitting the GitHub API
(which has tight unauthenticated rate limits and would break for guests).

Writes docs/manifest.json with shape:
    {
        "generated_at": "2026-04-25T11:03:14Z",
        "latest": "2026-04-25",
        "dates": ["2026-04-25", "2026-04-24", ...]   // newest first
    }

Also MIRRORS each {date}_filter_e.csv into docs/recommendations/. The
canonical home stays under data/output/recommendations/ (backtests, dashboards,
ad-hoc scripts all read from there), but GitHub Pages serves only the /docs
folder, so the mirror is what the live dashboard actually fetches.
"""
from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RECS_DIR = REPO_ROOT / "data" / "output" / "recommendations"
DOCS_DIR = REPO_ROOT / "docs"
DOCS_RECS_DIR = DOCS_DIR / "recommendations"

DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_filter_e\.csv$")


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_RECS_DIR.mkdir(parents=True, exist_ok=True)

    dates: list[str] = []
    if RECS_DIR.exists():
        for p in RECS_DIR.iterdir():
            m = DATE_RE.match(p.name)
            if m:
                dates.append(m.group(1))
                # Mirror into docs/ so the published Pages site can fetch it.
                # copy2 preserves mtime, which keeps git diffs minimal when
                # the file content hasn't changed across runs.
                shutil.copy2(p, DOCS_RECS_DIR / p.name)
    dates = sorted(set(dates), reverse=True)
    manifest = {
        "generated_at": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "latest": dates[0] if dates else None,
        "dates": dates,
    }
    out = DOCS_DIR / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"  wrote {out} ({len(dates)} date(s); latest={manifest['latest']}); "
        f"mirrored {len(dates)} CSV(s) into {DOCS_RECS_DIR}"
    )


if __name__ == "__main__":
    main()
