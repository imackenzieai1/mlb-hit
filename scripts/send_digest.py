#!/usr/bin/env python
"""Send the daily picks digest via Mailgun's HTTPS API.

Why Mailgun? It's the simplest secret-driven email path that works from a
GitHub Actions runner without SMTP credentials, app passwords, or OAuth
hand-shakes — just three secrets (API key, sending domain, recipient).

If you'd rather use SendGrid or Resend, the swap is one function below; the
contract (subject + html + optional attachment) stays identical.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests


def send_via_mailgun(
    api_key: str,
    domain: str,
    to_addr: str,
    from_addr: str,
    subject: str,
    html_body: str,
    attachment: Path | None,
) -> None:
    url = f"https://api.mailgun.net/v3/{domain}/messages"
    data = {
        "from": from_addr,
        "to": to_addr,
        "subject": subject,
        "html": html_body,
    }
    files = []
    if attachment and attachment.exists():
        files.append(("attachment", (attachment.name, attachment.read_bytes())))
    resp = requests.post(url, auth=("api", api_key), data=data, files=files, timeout=30)
    if resp.status_code >= 300:
        print(f"::error::Mailgun returned {resp.status_code}: {resp.text}")
        sys.exit(1)
    print(f"  email sent ({resp.status_code}): id={resp.json().get('id', '?')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--to", required=True)
    parser.add_argument("--from", dest="from_addr", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--html", required=True, help="Path to digest.html")
    parser.add_argument("--attach", default=None, help="Optional CSV to attach")
    args = parser.parse_args()

    api_key = os.environ.get("MAILGUN_API_KEY")
    domain = os.environ.get("MAILGUN_DOMAIN")
    if not api_key or not domain:
        print("  MAILGUN_API_KEY/MAILGUN_DOMAIN not set — skipping email.")
        return

    html_body = Path(args.html).read_text(encoding="utf-8")
    attachment = Path(args.attach) if args.attach else None

    send_via_mailgun(
        api_key=api_key,
        domain=domain,
        to_addr=args.to,
        from_addr=args.from_addr,
        subject=args.subject,
        html_body=html_body,
        attachment=attachment,
    )


if __name__ == "__main__":
    main()
