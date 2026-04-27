from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .config import SETTINGS

DATA = SETTINGS["paths"]["data_dir"]
DB = SETTINGS["paths"]["db_path"]


def raw_path(kind: str, name: str) -> Path:
    p = DATA / "raw" / kind
    p.mkdir(parents=True, exist_ok=True)
    return p / name


def clean_path(name: str) -> Path:
    p = DATA / "clean"
    p.mkdir(parents=True, exist_ok=True)
    return p / name


def modeling_path(name: str) -> Path:
    p = DATA / "modeling"
    p.mkdir(parents=True, exist_ok=True)
    return p / name


def output_path(kind: str, name: str) -> Path:
    p = DATA / "output" / kind
    p.mkdir(parents=True, exist_ok=True)
    return p / name


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, index=False)


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def get_db() -> sqlite3.Connection:
    DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db() -> None:
    conn = get_db()
    try:
        conn.executescript(
            """
        CREATE TABLE IF NOT EXISTS predictions (
            date TEXT, player_id INTEGER, player_name TEXT,
            team TEXT, opponent TEXT, lineup_spot INTEGER,
            p_model REAL, model_version TEXT, features_json TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (date, player_id, model_version)
        );
        CREATE TABLE IF NOT EXISTS recommendations (
            date TEXT, player_id INTEGER, book TEXT,
            price_american INTEGER, p_model REAL, edge REAL,
            stake_units REAL, created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS results (
            date TEXT, player_id INTEGER, pa INTEGER, hits INTEGER,
            got_hit INTEGER, graded_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (date, player_id)
        );
        """
        )
        conn.commit()
    finally:
        conn.close()
