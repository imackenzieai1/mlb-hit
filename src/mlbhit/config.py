from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "settings.yaml"


def load_settings() -> dict:
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    for k, v in cfg["paths"].items():
        cfg["paths"][k] = (REPO_ROOT / v).resolve()
        if k.endswith("_dir"):
            cfg["paths"][k].mkdir(parents=True, exist_ok=True)
    return cfg


def env(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key, default)


SETTINGS = load_settings()
