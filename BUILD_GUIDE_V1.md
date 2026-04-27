# MLB Hit-Prop V1 — Deep Build Guide

**Scope:** V1 focused on **player hits** (`P(≥1 hit)`). No home-run, no strikeout props. Just hits. Cursor + Mac + Python. Every player/pitcher with fewer than 50 AB (batters) or 100 BF (pitchers) is excluded from training and from daily predictions.

This guide walks you through the actual files, commands, and code. Copy-paste to run. Every module is kept small and testable. Where a decision could go either way, I pick one and explain why in a line or two.

---

## 0. Three Concept Answers You Asked For

### 0.1 What does that log-loss paragraph mean?

**Log loss** = the standard scoring rule for probability predictions. On a binary outcome `y ∈ {0, 1}` with predicted probability `p`:

```
loss_i = - ( y_i * log(p_i) + (1 - y_i) * log(1 - p_i) )
```

Average it across all predictions. Lower = better. Key properties:

- A perfectly calibrated model that actually matches reality gets a log loss equal to the entropy of the outcome (roughly 0.58–0.60 for hit-prop data).
- A model that always predicts the league base rate gets log loss ≈ 0.647.
- A model that confidently predicts 0.95 but is wrong gets hammered: `-log(0.05) ≈ 3.0`. Overconfidence is expensive.

So when I say "BA-only gets ~0.625 and a good v1 gets ~0.605–0.610":

- ~0.647: no-info constant prediction (baseline ceiling of badness).
- ~0.625: use only season BA → `1 - (1-BA)^4` as P(≥1 hit).
- ~0.610: a competent v1 with XGBoost, xBA, platoon splits, lineup spot, opposing pitcher xBA.
- ~0.600: realistic best-case before you start scraping lots of sources and bullpen-state data.
- ~0.58–0.59: theoretical floor given inherent randomness of baseball.

**Why 0.02 is huge:** the total "improvable" gap from 0.647 (no info) to 0.585 (floor) is only 0.062. Moving from 0.625 to 0.605 captures one-third of the total available signal. Betting-wise, that's enough to overcome sportsbook vig on hit props if your calibration is good.

**Bottom line:** measure log loss from day 1, beat 0.625, celebrate quietly; beat 0.610 and you have something worth betting.

### 0.2 Launch angle: manual weights vs letting the model decide

The question is "should I force a heavier weight on optimal launch angle, or can the model learn the weight on its own?"

**Recommendation: let the model learn, but engineer the features to make the signal obvious, and optionally use monotonic constraints to enforce direction.**

Why:
- XGBoost with 100k+ rows will learn the right weight far better than you can guess.
- But XGBoost will only learn about launch angle if you *feed* it launch-angle-derived features. It can't learn what it can't see.
- Monotonic constraints let you say "if feature X goes up, prediction must not decrease" without specifying the magnitude. This is the right way to inject a prior.

**Features to engineer from Statcast that carry launch-angle signal:**

| Feature | What it measures | Signal |
|---|---|---|
| `xBA` | Statcast's estimated BA given LA+EV (and sprint speed) | **Strongest single feature** — already internalizes LA |
| `sweet_spot_pct` | % batted balls with LA in [8°, 32°] | Direct LA signal for hits |
| `line_drive_pct` | Batted-ball classification | LDs fall in at ~68% |
| `mean_launch_angle` | Average LA on contact | Weak alone, useful with EV |
| `mean_exit_velocity` | Average EV on contact | Pairs with LA |
| `hard_hit_pct` | % batted balls ≥95 mph EV | Best contact-quality metric |
| `solid_contact_pct` | Statcast's "solid" + "barrel" + "flare/burner" | Covers hit-friendly contact |

These together capture the "optimal launch angle" story better than any single LA number.

**Monotonic constraints in XGBoost:**

```python
# Force: more sweet-spot % never decreases predicted P(hit)
# Force: more strikeouts never increases predicted P(hit)
monotone = {
    "sweet_spot_pct":       +1,
    "line_drive_pct":       +1,
    "hard_hit_pct":         +1,
    "xba_blend":            +1,
    "bat_k_pct":            -1,
    "sp_k_pct":             -1,
    "sp_xba_allowed":       +1,   # pitcher xBA up = more hits against = up
    "pen_xba_allowed":      +1,
    "exp_pa":               +1,
    "platoon_advantage":    +1,
}
# Other features unconstrained (0)
```

We'll wire this into the training module below. This is a cleaner way to "weight LA more" than hard-coding a coefficient, because it constrains the shape but not the magnitude.

### 0.3 The 50-AB rule (and why we pick 100 BF for pitchers)

**Rule applied everywhere:**
- Exclude any **batter with <50 season at-bats** from both training rows and daily prediction rows.
- Exclude any **pitcher with <100 batters faced** when computing opposing-pitcher features. Substitute a **league average** for that pitcher rather than dropping the whole game (otherwise you'd lose a ton of April prediction rows).

Why 50 AB for batters: below ~50 AB, xBA variance is too high to trust. BABIP has barely started regressing to skill. Your model can't learn clean patterns from noise.

Why 100 BF (not AB) for pitchers: pitchers' metrics are computed against all the batters they face. 100 BF is roughly 4–5 starts for an SP or 20–25 appearances for a RP. Below that, xBA-allowed is dominated by batted-ball luck.

For in-season daily use, April will have gaps. Fallback is league average for that pitcher's role (SP vs RP) and handedness. Flag with `pitcher_low_sample=1` so you can audit predictions for those games.

---

## 1. Mac + Cursor Setup (20 minutes)

If you already ran the Homebrew/pyenv part from the earlier plan, skip to 1.3.

### 1.1 System prerequisites

```bash
# Homebrew (skip if you already have it)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# pyenv + build deps
brew install pyenv openssl readline sqlite3 xz zlib tcl-tk

# Shell init (zsh)
cat >> ~/.zshrc <<'EOF'
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
EOF
source ~/.zshrc

# Python 3.11
pyenv install 3.11.9
pyenv global 3.11.9
python --version   # should show 3.11.9
```

### 1.2 Install Cursor

```bash
brew install --cask cursor
```

Open Cursor, sign in, and set your AI preferences. Cursor is a VS Code fork, so everything that works in VS Code works here (same keybindings, same extensions marketplace).

### 1.3 Project folder + venv

```bash
mkdir -p ~/projects/mlb-hitprop && cd ~/projects/mlb-hitprop
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 1.4 Requirements file

Create `requirements.txt`:

```
pandas>=2.2
numpy>=1.26
scipy>=1.13
scikit-learn>=1.5
xgboost>=2.0
pybaseball>=2.2
MLB-StatsAPI>=1.7
requests>=2.32
python-dotenv>=1.0
pyarrow>=16.0
joblib>=1.4
pyyaml>=6.0
tqdm>=4.66
matplotlib>=3.9
streamlit>=1.35
ipykernel>=6.29
jupyterlab>=4.2
```

Install:

```bash
pip install -r requirements.txt
python -m ipykernel install --user --name mlb-hitprop --display-name "Python (mlb-hitprop)"
```

### 1.5 Open in Cursor, bind the interpreter

```bash
cursor .
```

In Cursor: `Cmd+Shift+P` → "Python: Select Interpreter" → pick `~/projects/mlb-hitprop/.venv/bin/python`.

Install two extensions (Extensions pane, left bar):
- **Python** (Microsoft)
- **Jupyter** (Microsoft)

Test the kernel by creating a scratch notebook and running `import pandas; import xgboost; print("ok")`.

### 1.6 Environment variables and gitignore

Create `.env.example`:

```
ODDS_API_KEY=
OPENWEATHER_API_KEY=
TZ=America/New_York
DATA_DIR=./data
DB_PATH=./db/tracking.sqlite
```

Copy to `.env` and fill in your API keys:

```bash
cp .env.example .env
# edit .env in Cursor, paste keys
```

Get the keys:
- **OpenWeatherMap:** https://openweathermap.org/api — free tier, instant email verification.
- **The Odds API:** https://the-odds-api.com/ — free tier is enough for game lines, paid tier required later for player props.

Create `.gitignore`:

```
.venv/
.env
__pycache__/
*.pyc
data/raw/
data/clean/
data/modeling/
data/output/
db/
logs/
models/*.joblib
.ipynb_checkpoints/
.DS_Store
```

Initialize git:

```bash
git init
git add .gitignore requirements.txt .env.example
git commit -m "initial project skeleton"
```

---

## 2. Folder Skeleton

Create the tree:

```bash
mkdir -p config data/{raw,clean,modeling,output} db logs models notebooks \
         src/mlbhit/{pipeline,features,model,utils} tests app
touch src/mlbhit/__init__.py \
      src/mlbhit/pipeline/__init__.py \
      src/mlbhit/features/__init__.py \
      src/mlbhit/model/__init__.py \
      src/mlbhit/utils/__init__.py
```

Also make the package importable from the repo root by adding a `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "mlbhit"
version = "0.1.0"
requires-python = ">=3.11"

[tool.setuptools.packages.find]
where = ["src"]
```

Then install in editable mode:

```bash
pip install -e .
```

Now `from mlbhit.utils.odds_math import prob_to_american` works anywhere.

---

## 3. Config Layer

### 3.1 `config/settings.yaml`

```yaml
paths:
  data_dir: data
  db_path: db/tracking.sqlite
  models_dir: models

filters:
  min_batter_ab: 50
  min_pitcher_bf: 100

seasons:
  train: [2023, 2024]
  validation: [2025]

rolling:
  windows_days: [14, 30]

blend_weights:
  season: 0.7
  recent: 0.3

edge_threshold: 0.05
kelly_fraction: 0.0    # v1: flat-stake. Bump to 0.25 later.
```

### 3.2 `src/mlbhit/config.py`

```python
from __future__ import annotations
from pathlib import Path
import os
import yaml
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "settings.yaml"

def load_settings() -> dict:
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    # resolve paths relative to repo root
    for k, v in cfg["paths"].items():
        cfg["paths"][k] = (REPO_ROOT / v).resolve()
        if k.endswith("_dir"):
            cfg["paths"][k].mkdir(parents=True, exist_ok=True)
    return cfg

def env(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key, default)

SETTINGS = load_settings()
```

Test:

```bash
python -c "from mlbhit.config import SETTINGS; print(SETTINGS)"
```

---

## 4. Utility Modules

### 4.1 `src/mlbhit/utils/odds_math.py`

```python
from __future__ import annotations
import numpy as np

def prob_to_american(p: float) -> int:
    p = max(min(p, 0.9999), 0.0001)
    if p >= 0.5:
        return int(round(-100 * p / (1 - p)))
    return int(round(100 * (1 - p) / p))

def american_to_prob(odds: int) -> float:
    if odds > 0:
        return 100 / (odds + 100)
    return -odds / (-odds + 100)

def american_to_decimal(odds: int) -> float:
    if odds > 0:
        return 1 + odds / 100
    return 1 + 100 / -odds

def devig_two_way(p_a: float, p_b: float) -> tuple[float, float]:
    s = p_a + p_b
    return p_a / s, p_b / s

def ev_per_unit(p_model: float, odds_american: int) -> float:
    """Expected $ return per $1 wagered at given American odds."""
    payout = odds_american / 100 if odds_american > 0 else 100 / -odds_american
    return p_model * payout - (1 - p_model)

def kelly_fraction(p: float, odds_american: int) -> float:
    d = american_to_decimal(odds_american)
    b = d - 1
    q = 1 - p
    return max(0.0, (b * p - q) / b) if b > 0 else 0.0
```

### 4.2 `src/mlbhit/utils/dates.py`

```python
from __future__ import annotations
from datetime import date, datetime, timedelta
import pytz

ET = pytz.timezone("America/New_York")

def today_et() -> date:
    return datetime.now(ET).date()

def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)

def ymd(d: date) -> str:
    return d.strftime("%Y-%m-%d")
```

### 4.3 `src/mlbhit/utils/ids.py`

```python
from __future__ import annotations
import re
import pandas as pd
from pybaseball import playerid_lookup

def normalize_name(name: str) -> str:
    n = name.strip()
    n = re.sub(r"\s+(Jr\.?|Sr\.?|II|III|IV)$", "", n, flags=re.IGNORECASE)
    n = n.replace(".", "").replace(",", "")
    return n.lower()

def lookup_mlbam(first: str, last: str) -> int | None:
    try:
        df = playerid_lookup(last, first, fuzzy=True)
        if df.empty:
            return None
        active = df.sort_values("mlb_played_last", ascending=False).iloc[0]
        return int(active["key_mlbam"])
    except Exception:
        return None
```

### 4.4 `src/mlbhit/io.py`

```python
from __future__ import annotations
from pathlib import Path
import pandas as pd
import sqlite3
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
    with get_db() as conn:
        conn.executescript("""
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
        """)
```

Initialize once:

```bash
python -c "from mlbhit.io import init_db; init_db(); print('db ready')"
```

---

## 5. Reference Data: Stadiums

Create `config/stadiums.csv` (minimum starter set; fill in the other teams the same way):

```csv
team,venue,lat,lon,is_dome,is_retractable,park_factor,singles_park_factor,orientation_deg
ARI,Chase Field,33.4455,-112.0667,0,1,103,101,24
ATL,Truist Park,33.8908,-84.4678,0,0,101,100,55
BAL,Oriole Park at Camden Yards,39.2838,-76.6217,0,0,101,101,32
BOS,Fenway Park,42.3467,-71.0972,0,0,104,108,44
CHC,Wrigley Field,41.9484,-87.6553,0,0,103,102,39
CHW,Guaranteed Rate Field,41.8300,-87.6339,0,0,98,99,42
CIN,Great American Ball Park,39.0976,-84.5068,0,0,106,103,41
CLE,Progressive Field,41.4959,-81.6852,0,0,99,100,42
COL,Coors Field,39.7559,-104.9942,0,0,112,115,0
DET,Comerica Park,42.3390,-83.0485,0,0,97,99,42
HOU,Minute Maid Park,29.7572,-95.3557,0,1,100,101,350
KC,Kauffman Stadium,39.0517,-94.4803,0,0,98,99,46
LAA,Angel Stadium,33.8003,-117.8827,0,0,99,100,50
LAD,Dodger Stadium,34.0739,-118.2400,0,0,99,99,25
MIA,loanDepot park,25.7781,-80.2197,0,1,95,97,40
MIL,American Family Field,43.0280,-87.9712,0,1,100,100,55
MIN,Target Field,44.9817,-93.2776,0,0,100,102,90
NYM,Citi Field,40.7571,-73.8458,0,0,97,98,28
NYY,Yankee Stadium,40.8296,-73.9262,0,0,102,101,31
OAK,Oakland Coliseum,37.7516,-122.2005,0,0,94,97,55
PHI,Citizens Bank Park,39.9061,-75.1665,0,0,103,100,40
PIT,PNC Park,40.4469,-80.0057,0,0,98,101,55
SD,Petco Park,32.7073,-117.1570,0,0,97,99,20
SF,Oracle Park,37.7786,-122.3893,0,0,94,97,90
SEA,T-Mobile Park,47.5914,-122.3325,0,1,97,99,2
STL,Busch Stadium,38.6226,-90.1928,0,0,98,100,60
TB,Tropicana Field,27.7682,-82.6534,1,0,96,99,45
TEX,Globe Life Field,32.7473,-97.0817,0,1,103,102,30
TOR,Rogers Centre,43.6414,-79.3894,0,1,101,100,0
WSH,Nationals Park,38.8730,-77.0074,0,0,100,101,30
```

(Park factors are rough 100-centered placeholders. Update from FanGraphs' published park factors for accuracy. `orientation_deg` is the direction from home plate to center field in compass degrees.)

Also create `config/pa_lookup.csv`:

```csv
lineup_spot,home_away,exp_pa
1,A,4.60
2,A,4.50
3,A,4.40
4,A,4.30
5,A,4.20
6,A,4.10
7,A,4.00
8,A,3.90
9,A,3.80
1,H,4.55
2,H,4.45
3,H,4.35
4,H,4.25
5,H,4.15
6,H,4.05
7,H,3.95
8,H,3.85
9,H,3.75
```

---

## 6. Fetching Modules

### 6.1 `src/mlbhit/pipeline/fetch_schedule.py`

```python
from __future__ import annotations
from datetime import date
import pandas as pd
import statsapi
from ..io import raw_path

def fetch_schedule(d: date) -> pd.DataFrame:
    games = statsapi.schedule(date=d.strftime("%m/%d/%Y"))
    rows = []
    for g in games:
        rows.append({
            "date": d.isoformat(),
            "game_pk": g["game_id"],
            "status": g["status"],
            "home_team": g["home_name"],
            "away_team": g["away_name"],
            "home_abbr": g.get("home_id"),  # we'll map later
            "away_abbr": g.get("away_id"),
            "venue": g.get("venue_name"),
            "game_datetime": g.get("game_datetime"),
            "home_probable_pitcher": g.get("home_probable_pitcher"),
            "away_probable_pitcher": g.get("away_probable_pitcher"),
            "home_probable_pitcher_id": g.get("home_probable_pitcher_id"),
            "away_probable_pitcher_id": g.get("away_probable_pitcher_id"),
        })
    df = pd.DataFrame(rows)
    out = raw_path("schedule", f"{d.isoformat()}.parquet")
    df.to_parquet(out, index=False)
    return df

if __name__ == "__main__":
    from datetime import date
    df = fetch_schedule(date.today())
    print(df.head())
    print(f"{len(df)} games")
```

Run:

```bash
python -m mlbhit.pipeline.fetch_schedule
```

### 6.2 `src/mlbhit/pipeline/fetch_batting_stats.py`

Applies the 50-AB filter here.

```python
from __future__ import annotations
import pandas as pd
from pybaseball import batting_stats
from ..io import clean_path
from ..config import SETTINGS

MIN_AB = SETTINGS["filters"]["min_batter_ab"]

# Columns we want. pybaseball returns many; we pick the essentials.
KEEP = [
    "IDfg", "Name", "Team", "Age", "G", "AB", "PA", "H",
    "AVG", "OBP", "SLG",
    "BB%", "K%",
    "Hard%", "LD%", "GB%", "FB%", "IFFB%",
    "Contact%", "Z-Contact%", "O-Swing%",
    "wRC+", "Bat",
    # Statcast-derived (available from FanGraphs in recent seasons)
    "xBA", "xwOBA", "Barrel%", "HardHit%", "maxEV", "EV", "LA",
]

def fetch_batting(season: int) -> pd.DataFrame:
    df = batting_stats(season, qual=0)  # qual=0 to let us filter ourselves
    # keep only columns we know exist
    existing = [c for c in KEEP if c in df.columns]
    df = df[existing].copy()
    # apply 50-AB filter
    df = df[df["AB"] >= MIN_AB].reset_index(drop=True)
    df["season"] = season
    df = df.rename(columns={
        "IDfg": "fg_id",
        "Name": "player_name",
        "Team": "team",
        "AVG": "ba",
        "OBP": "obp",
        "SLG": "slg",
        "BB%": "bb_pct",
        "K%": "k_pct",
        "Hard%": "hard_pct_fg",
        "LD%": "ld_pct",
        "GB%": "gb_pct",
        "FB%": "fb_pct",
        "Contact%": "contact_pct",
        "Z-Contact%": "z_contact_pct",
        "O-Swing%": "chase_pct",
        "xBA": "xba",
        "xwOBA": "xwoba",
        "Barrel%": "barrel_pct",
        "HardHit%": "hard_hit_pct",
        "maxEV": "max_ev",
        "EV": "mean_ev",
        "LA": "mean_la",
    })
    # convert "22.5 %" strings → floats if needed
    for c in ["bb_pct","k_pct","hard_pct_fg","ld_pct","gb_pct","fb_pct",
              "contact_pct","z_contact_pct","chase_pct",
              "barrel_pct","hard_hit_pct"]:
        if c in df.columns and df[c].dtype == object:
            df[c] = df[c].astype(str).str.replace("%", "").astype(float) / 100.0
    return df

def save_batting(seasons: list[int]) -> pd.DataFrame:
    frames = [fetch_batting(s) for s in seasons]
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(clean_path("batter_season_stats.parquet"), index=False)
    return df

if __name__ == "__main__":
    df = save_batting([2023, 2024])
    print(f"{len(df)} batter-seasons kept after AB>={MIN_AB} filter")
    print(df[["player_name","team","season","AB","ba","xba","hard_hit_pct"]].head())
```

Run:

```bash
python -m mlbhit.pipeline.fetch_batting_stats
```

**Where this will fail in daily use:** pybaseball occasionally 403s or returns empty for the current season in early April. Handle by retrying once after 5 seconds, then falling back to last-saved parquet.

### 6.3 `src/mlbhit/pipeline/fetch_pitching_stats.py`

```python
from __future__ import annotations
import pandas as pd
from pybaseball import pitching_stats
from ..io import clean_path
from ..config import SETTINGS

MIN_BF = SETTINGS["filters"]["min_pitcher_bf"]

KEEP = [
    "IDfg","Name","Team","Age","G","GS","IP","TBF","ERA","FIP","xFIP",
    "K%","BB%","WHIP","AVG","GB%","LD%","HR/9",
    "HardHit%","Barrel%","xBA","xwOBA","LA","EV",
    "vFA (pi)","vFA (sc)",  # fastball velo approximations
]

def fetch_pitching(season: int) -> pd.DataFrame:
    df = pitching_stats(season, qual=0)
    existing = [c for c in KEEP if c in df.columns]
    df = df[existing].copy()
    # TBF = batters faced
    df = df[df["TBF"] >= MIN_BF].reset_index(drop=True)
    df["season"] = season
    df = df.rename(columns={
        "IDfg":"fg_id",
        "Name":"pitcher_name",
        "Team":"team",
        "K%":"k_pct_allowed",
        "BB%":"bb_pct_allowed",
        "AVG":"ba_allowed",
        "GB%":"gb_pct_allowed",
        "LD%":"ld_pct_allowed",
        "HardHit%":"hard_hit_pct_allowed",
        "Barrel%":"barrel_pct_allowed",
        "xBA":"xba_allowed",
        "xwOBA":"xwoba_allowed",
    })
    for c in ["k_pct_allowed","bb_pct_allowed","gb_pct_allowed","ld_pct_allowed",
              "hard_hit_pct_allowed","barrel_pct_allowed"]:
        if c in df.columns and df[c].dtype == object:
            df[c] = df[c].astype(str).str.replace("%","").astype(float) / 100.0
    df["role"] = (df["GS"] / df["G"].clip(lower=1)).apply(
        lambda r: "SP" if r >= 0.6 else "RP"
    )
    return df

def save_pitching(seasons: list[int]) -> pd.DataFrame:
    frames = [fetch_pitching(s) for s in seasons]
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(clean_path("pitcher_season_stats.parquet"), index=False)
    return df

if __name__ == "__main__":
    df = save_pitching([2023, 2024])
    print(f"{len(df)} pitcher-seasons kept after BF>={MIN_BF} filter")
    print(df[["pitcher_name","team","season","TBF","xba_allowed","k_pct_allowed","role"]].head())
```

### 6.4 `src/mlbhit/pipeline/fetch_statcast.py`

Statcast pulls the launch-angle-derived features (sweet-spot %, etc.) that the FanGraphs tables don't always surface in a clean way. This is slow — plan on 10 min for a full season pull.

```python
from __future__ import annotations
import pandas as pd
from pybaseball import statcast_batter_expected_stats, statcast
from datetime import date
from ..io import clean_path, raw_path

def pull_statcast_season(season: int) -> pd.DataFrame:
    start = f"{season}-03-20"
    end = f"{season}-11-05"
    df = statcast(start_dt=start, end_dt=end)
    out = raw_path("statcast", f"pitches_{season}.parquet")
    df.to_parquet(out, index=False)
    return df

def derive_batter_la_features(pitches: pd.DataFrame) -> pd.DataFrame:
    """Per batter per season: sweet-spot %, LD%, mean LA, mean EV, hard-hit %."""
    bb = pitches[pitches["type"] == "X"].copy()  # batted balls
    bb["is_sweet_spot"] = bb["launch_angle"].between(8, 32)
    bb["is_hard_hit"] = bb["launch_speed"] >= 95
    bb["is_line_drive"] = bb["bb_type"] == "line_drive"
    bb["is_solid"] = bb["launch_speed_angle"].isin([4, 5, 6])  # flare/burner/solid/barrel
    agg = bb.groupby(["batter", "game_year"]).agg(
        sweet_spot_pct=("is_sweet_spot", "mean"),
        line_drive_pct=("is_line_drive", "mean"),
        mean_launch_angle=("launch_angle", "mean"),
        mean_exit_velocity=("launch_speed", "mean"),
        hard_hit_pct=("is_hard_hit", "mean"),
        solid_contact_pct=("is_solid", "mean"),
        batted_balls=("is_sweet_spot", "size"),
    ).reset_index()
    agg = agg.rename(columns={"batter": "mlbam_id", "game_year": "season"})
    return agg

def derive_pitcher_la_features(pitches: pd.DataFrame) -> pd.DataFrame:
    bb = pitches[pitches["type"] == "X"].copy()
    bb["is_sweet_spot"] = bb["launch_angle"].between(8, 32)
    bb["is_hard_hit"] = bb["launch_speed"] >= 95
    bb["is_line_drive"] = bb["bb_type"] == "line_drive"
    agg = bb.groupby(["pitcher", "game_year"]).agg(
        sweet_spot_pct_allowed=("is_sweet_spot", "mean"),
        line_drive_pct_allowed=("is_line_drive", "mean"),
        hard_hit_pct_allowed=("is_hard_hit", "mean"),
        batted_balls_allowed=("is_sweet_spot", "size"),
    ).reset_index()
    agg = agg.rename(columns={"pitcher": "mlbam_id", "game_year": "season"})
    return agg

if __name__ == "__main__":
    for s in [2023, 2024]:
        print(f"pulling {s}…")
        pitches = pull_statcast_season(s)
        bat_la = derive_batter_la_features(pitches)
        pit_la = derive_pitcher_la_features(pitches)
        bat_la.to_parquet(clean_path(f"batter_la_{s}.parquet"), index=False)
        pit_la.to_parquet(clean_path(f"pitcher_la_{s}.parquet"), index=False)
        print(f"  {len(bat_la)} batters, {len(pit_la)} pitchers")
```

### 6.5 `src/mlbhit/pipeline/fetch_boxscores.py`

Gets the training target. For every game in a date range, pulls every batter's PA and hits.

```python
from __future__ import annotations
from datetime import date
from typing import Iterable
import pandas as pd
import statsapi
from tqdm import tqdm
from ..io import clean_path
from ..utils.dates import daterange

def fetch_day(d: date) -> list[dict]:
    games = statsapi.schedule(date=d.strftime("%m/%d/%Y"))
    rows = []
    for g in games:
        if g["status"] not in ("Final", "Game Over", "Completed Early"):
            continue
        try:
            box = statsapi.boxscore_data(g["game_id"])
        except Exception:
            continue
        for side in ("home", "away"):
            players = box.get(side, {}).get("players", {})
            team = box["teamInfo"][side]["abbreviation"]
            opp_side = "away" if side == "home" else "home"
            opp = box["teamInfo"][opp_side]["abbreviation"]
            venue = g.get("venue_name")
            for pid, p in players.items():
                stats = p.get("stats", {}).get("batting", {})
                if not stats:
                    continue
                ab = int(stats.get("atBats", 0))
                pa = int(stats.get("plateAppearances", 0) or 0)
                hits = int(stats.get("hits", 0))
                if pa == 0:
                    continue
                rows.append({
                    "date": d.isoformat(),
                    "game_pk": g["game_id"],
                    "player_id": int(p["person"]["id"]),
                    "player_name": p["person"]["fullName"],
                    "team": team,
                    "opponent": opp,
                    "home_away": "H" if side == "home" else "A",
                    "venue": venue,
                    "batting_order": int(p.get("battingOrder", "0")) // 100 or None,
                    "ab": ab,
                    "pa": pa,
                    "hits": hits,
                    "got_hit": int(hits > 0),
                })
    return rows

def fetch_range(start: date, end: date) -> pd.DataFrame:
    all_rows = []
    for d in tqdm(list(daterange(start, end))):
        all_rows.extend(fetch_day(d))
    return pd.DataFrame(all_rows)

if __name__ == "__main__":
    df = fetch_range(date(2024, 4, 1), date(2024, 9, 29))
    df.to_parquet(clean_path("boxscores_2024.parquet"), index=False)
    print(df.shape, df["got_hit"].mean())
```

**This one takes ~20–40 minutes for a full season** because of the API rate. Run it once per season and cache.

### 6.6 `src/mlbhit/pipeline/fetch_lineups.py`

```python
from __future__ import annotations
from datetime import date
import pandas as pd
import statsapi
from ..io import raw_path

def fetch_lineups(d: date) -> pd.DataFrame:
    games = statsapi.schedule(date=d.strftime("%m/%d/%Y"))
    rows = []
    for g in games:
        pk = g["game_id"]
        try:
            box = statsapi.boxscore_data(pk)
        except Exception:
            continue
        for side in ("home","away"):
            team = box["teamInfo"][side]["abbreviation"]
            opp_side = "away" if side=="home" else "home"
            opp = box["teamInfo"][opp_side]["abbreviation"]
            order = box[side].get("battingOrder", [])  # list of player IDs in order
            for spot, pid in enumerate(order, start=1):
                pid = int(pid)
                player = box[side]["players"].get(f"ID{pid}", {})
                name = player.get("person", {}).get("fullName", "")
                rows.append({
                    "date": d.isoformat(),
                    "game_pk": pk,
                    "team": team,
                    "opponent": opp,
                    "home_away": "H" if side=="home" else "A",
                    "player_id": pid,
                    "player_name": name,
                    "lineup_spot": spot,
                    "lineup_confirmed": bool(order),
                })
    df = pd.DataFrame(rows)
    out = raw_path("lineups", f"{d.isoformat()}.parquet")
    df.to_parquet(out, index=False)
    return df

if __name__ == "__main__":
    from datetime import date
    df = fetch_lineups(date.today())
    print(df.head())
    print(f"{df['lineup_confirmed'].sum()} confirmed lineup slots")
```

### 6.7 `src/mlbhit/pipeline/fetch_weather.py`

```python
from __future__ import annotations
import requests
import pandas as pd
from datetime import datetime
from ..io import raw_path
from ..config import env

OWM = "https://api.openweathermap.org/data/2.5/weather"

def fetch_weather_for_coords(lat: float, lon: float) -> dict:
    key = env("OPENWEATHER_API_KEY")
    r = requests.get(OWM, params={"lat": lat, "lon": lon, "units": "imperial", "appid": key}, timeout=10)
    r.raise_for_status()
    j = r.json()
    return {
        "temp_f": j["main"]["temp"],
        "wind_mph": j["wind"]["speed"],
        "wind_deg": j["wind"].get("deg", 0),
        "humidity": j["main"]["humidity"],
        "conditions": j["weather"][0]["main"],
        "precip_flag": 1 if j["weather"][0]["main"] in ("Rain","Drizzle","Thunderstorm","Snow") else 0,
    }

def signed_wind(wind_mph: float, wind_deg: float, stadium_orientation_deg: float) -> float:
    """Positive if wind blowing toward center field (helpful), negative if blowing in."""
    import math
    # angle between wind direction and stadium's CF orientation
    delta = abs(((wind_deg - stadium_orientation_deg + 180) % 360) - 180)
    # 0° = dead out, 180° = dead in
    factor = math.cos(math.radians(delta))  # +1 out, -1 in
    return wind_mph * factor

if __name__ == "__main__":
    # Test with Fenway
    w = fetch_weather_for_coords(42.3467, -71.0972)
    print(w)
```

### 6.8 `src/mlbhit/pipeline/fetch_odds.py`

```python
from __future__ import annotations
import requests
import pandas as pd
from datetime import datetime
from ..io import raw_path
from ..config import env

ODDS = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"

def fetch_game_odds() -> pd.DataFrame:
    key = env("ODDS_API_KEY")
    params = {
        "apiKey": key,
        "regions": "us",
        "markets": "h2h,totals",
        "oddsFormat": "american",
    }
    r = requests.get(ODDS, params=params, timeout=10)
    r.raise_for_status()
    rows = []
    for game in r.json():
        for book in game.get("bookmakers", []):
            for market in book.get("markets", []):
                for o in market.get("outcomes", []):
                    rows.append({
                        "fetched_at": datetime.utcnow().isoformat(),
                        "commence_time": game["commence_time"],
                        "home_team": game["home_team"],
                        "away_team": game["away_team"],
                        "book": book["key"],
                        "market": market["key"],
                        "name": o["name"],
                        "price": o["price"],
                        "point": o.get("point"),
                    })
    df = pd.DataFrame(rows)
    if not df.empty:
        out = raw_path("odds", f"{datetime.utcnow().date().isoformat()}_{datetime.utcnow().strftime('%H%M')}.parquet")
        df.to_parquet(out, index=False)
    return df

if __name__ == "__main__":
    df = fetch_game_odds()
    print(df.head())
```

---

## 7. Feature Modules

### 7.1 `src/mlbhit/features/batter.py`

Build a single row per (batter, date) with blended season + recent features.

```python
from __future__ import annotations
import pandas as pd
import numpy as np
from ..io import clean_path
from ..config import SETTINGS

BLEND_S = SETTINGS["blend_weights"]["season"]
BLEND_R = SETTINGS["blend_weights"]["recent"]

def regress(x: pd.Series, n: pd.Series, prior: float, n_prior: int = 100) -> pd.Series:
    return (x * n + prior * n_prior) / (n + n_prior)

def build_batter_features(season: int) -> pd.DataFrame:
    bat = pd.read_parquet(clean_path("batter_season_stats.parquet"))
    bat = bat[bat["season"] == season].copy()
    la = pd.read_parquet(clean_path(f"batter_la_{season}.parquet"))
    # attach statcast LA features via mlbam_id → need a crosswalk
    # (For now, assume we have mlbam_id on bat — you'll build the ID map in §8.)
    bat = bat.merge(la, on=["mlbam_id","season"], how="left")

    # regress xBA to league mean with a prior of 100 PA
    league_xba = bat["xba"].median()
    bat["xba_regressed"] = regress(bat["xba"], bat["AB"], league_xba, n_prior=100)
    bat["k_pct_regressed"] = regress(bat["k_pct"], bat["PA"], bat["k_pct"].median(), n_prior=150)

    # column rename for consistency
    bat = bat.rename(columns={
        "xba_regressed":"bat_xba_season",
        "k_pct_regressed":"bat_k_pct",
        "hard_hit_pct":"bat_hard_hit_pct",
        "contact_pct":"bat_contact_pct",
        "sweet_spot_pct":"bat_sweet_spot_pct",
        "line_drive_pct":"bat_line_drive_pct",
        "solid_contact_pct":"bat_solid_contact_pct",
    })
    keep = ["mlbam_id","player_name","team","season",
            "bat_xba_season","bat_k_pct","bat_hard_hit_pct","bat_contact_pct",
            "bat_sweet_spot_pct","bat_line_drive_pct","bat_solid_contact_pct","AB","PA"]
    return bat[[c for c in keep if c in bat.columns]]
```

### 7.2 `src/mlbhit/features/pitcher.py`

```python
from __future__ import annotations
import pandas as pd
from ..io import clean_path

def build_pitcher_features(season: int) -> pd.DataFrame:
    pit = pd.read_parquet(clean_path("pitcher_season_stats.parquet"))
    pit = pit[pit["season"] == season].copy()
    la = pd.read_parquet(clean_path(f"pitcher_la_{season}.parquet"))
    pit = pit.merge(la, on=["mlbam_id","season"], how="left")
    pit = pit.rename(columns={
        "xba_allowed":"sp_xba_allowed",
        "k_pct_allowed":"sp_k_pct",
        "hard_hit_pct_allowed":"sp_hard_hit_allowed",
        "sweet_spot_pct_allowed":"sp_sweet_spot_allowed",
    })
    keep = ["mlbam_id","pitcher_name","team","season","role",
            "sp_xba_allowed","sp_k_pct","sp_hard_hit_allowed","sp_sweet_spot_allowed","IP","TBF"]
    return pit[[c for c in keep if c in pit.columns]]
```

### 7.3 `src/mlbhit/features/pa.py`

```python
from __future__ import annotations
import pandas as pd
from ..config import REPO_ROOT if False else None  # keep import clean

def load_pa_lookup() -> pd.DataFrame:
    from ..config import REPO_ROOT
    return pd.read_csv(REPO_ROOT / "config" / "pa_lookup.csv")

def expected_pa(lineup_spot: int, home_away: str, game_total: float | None = None) -> float:
    tbl = load_pa_lookup()
    base = tbl[(tbl["lineup_spot"] == lineup_spot) & (tbl["home_away"] == home_away)]["exp_pa"].iloc[0]
    if game_total is not None:
        base += 0.10 * ((game_total - 8.5) / 0.5)
    return float(max(3.0, min(5.5, base)))
```

### 7.4 `src/mlbhit/features/park_weather.py`

```python
from __future__ import annotations
import pandas as pd
from ..config import REPO_ROOT

def load_stadiums() -> pd.DataFrame:
    return pd.read_csv(REPO_ROOT / "config" / "stadiums.csv")

def attach_park(df: pd.DataFrame) -> pd.DataFrame:
    s = load_stadiums()
    return df.merge(s, left_on="home_team", right_on="team", how="left", suffixes=("","_park"))
```

---

## 8. Player ID Crosswalk

pybaseball returns FanGraphs IDs; MLB Stats API returns MLBAM IDs. You need both tied together.

### 8.1 `src/mlbhit/pipeline/build_player_map.py`

```python
from __future__ import annotations
import pandas as pd
from pybaseball import chadwick_register
from ..io import clean_path

def build() -> pd.DataFrame:
    reg = chadwick_register()
    df = reg[["key_mlbam","key_fangraphs","key_retro","name_first","name_last","mlb_played_first","mlb_played_last"]].dropna(subset=["key_mlbam"])
    df = df[df["mlb_played_last"].fillna(0) >= 2020]
    df["player_name"] = df["name_first"].str.strip() + " " + df["name_last"].str.strip()
    df = df.rename(columns={"key_mlbam":"mlbam_id","key_fangraphs":"fg_id"})
    df.to_parquet(clean_path("players.parquet"), index=False)
    return df

if __name__ == "__main__":
    df = build()
    print(f"{len(df)} players mapped")
```

Then in your batter/pitcher features, merge by `fg_id` to get `mlbam_id` for joins to boxscores/lineups/Statcast.

---

## 9. Build the Modeling Table

This is the big join.

### 9.1 `src/mlbhit/pipeline/build_features.py`

```python
from __future__ import annotations
import pandas as pd
import numpy as np
from ..io import clean_path, modeling_path
from ..features.batter import build_batter_features
from ..features.pitcher import build_pitcher_features
from ..features.park_weather import attach_park
from ..features.pa import expected_pa

def build_modeling_table(training_seasons: list[int]) -> pd.DataFrame:
    # boxscores give us one row per player-game with target
    frames = []
    for s in training_seasons:
        try:
            b = pd.read_parquet(clean_path(f"boxscores_{s}.parquet"))
            b["season"] = s
            frames.append(b)
        except FileNotFoundError:
            continue
    box = pd.concat(frames, ignore_index=True)

    # attach player map to be sure
    players = pd.read_parquet(clean_path("players.parquet"))[["mlbam_id","fg_id"]]
    box = box.merge(players, left_on="player_id", right_on="mlbam_id", how="left")

    # batter features (season-level, lagged by using prior-season stats and/or early-season regression)
    all_bat = pd.concat([build_batter_features(s) for s in training_seasons], ignore_index=True)
    box = box.merge(all_bat, on=["mlbam_id","season"], how="inner",
                    suffixes=("","_bat"))
    # INNER JOIN enforces the 50-AB filter because low-AB batters aren't in all_bat

    # pitcher features: we need the opposing starter per game
    # Join box to schedule to get probable starter, then to pitcher features
    schedules = []
    for s in training_seasons:
        try:
            import glob
            for path in sorted((clean_path("").parent / "raw" / "schedule").glob(f"{s}-*.parquet")):
                schedules.append(pd.read_parquet(path))
        except Exception:
            pass
    if schedules:
        sched = pd.concat(schedules, ignore_index=True)
        # For simplicity: use probable pitcher fields
        # Attach to box by game_pk + batting team (opposing pitcher is the other team's starter)
        box = box.merge(
            sched[["game_pk","home_probable_pitcher_id","away_probable_pitcher_id"]],
            on="game_pk", how="left",
        )
        box["opp_sp_id"] = np.where(
            box["home_away"] == "H",
            box["away_probable_pitcher_id"],
            box["home_probable_pitcher_id"],
        )
    else:
        box["opp_sp_id"] = np.nan

    all_pit = pd.concat([build_pitcher_features(s) for s in training_seasons], ignore_index=True)
    box = box.merge(
        all_pit.add_prefix("sp_"),
        left_on=["opp_sp_id","season"], right_on=["sp_mlbam_id","sp_season"],
        how="left",
    )
    # For pitchers below 100 BF (NOT in all_pit), fill with league mean
    league = all_pit[["sp_xba_allowed","sp_k_pct","sp_hard_hit_allowed","sp_sweet_spot_allowed"]].mean()
    for c, v in league.items():
        box[c] = box[c].fillna(v)
    box["pitcher_low_sample"] = box["sp_mlbam_id"].isna().astype(int)

    # bullpen: team-level rolling xBA-allowed of relievers.
    # v1: just league-average pen xBA; we'll refine in §10.
    box["pen_xba_allowed"] = league["sp_xba_allowed"]  # placeholder

    # platoon advantage: need bat_side + sp_throws.
    # (assume players.parquet has handedness columns — enrich from MLB Stats API if needed)
    # v1 placeholder: unknown platoon → 0
    box["platoon_advantage"] = 0

    # expected PA
    box["exp_pa"] = box.apply(
        lambda r: expected_pa(int(r["batting_order"]) if pd.notna(r["batting_order"]) else 5,
                              r["home_away"]), axis=1)

    # park factor
    box = attach_park(box)

    # derived features
    box["xba_diff"] = box["bat_xba_season"] - box["sp_xba_allowed"]
    box["exposure_wtd_opp_xba"] = 0.7 * box["sp_xba_allowed"] + 0.3 * box["pen_xba_allowed"]

    out = modeling_path("player_game_features.parquet")
    box.to_parquet(out, index=False)
    return box

if __name__ == "__main__":
    df = build_modeling_table([2023, 2024])
    print(df.shape)
    print(df.columns.tolist())
    print("hit rate:", df["got_hit"].mean())
```

You'll iterate on this several times before it's clean. That's expected.

---

## 10. Bullpen Features (v1-lite)

Rather than scraping bullpen leaderboards, derive from pitcher stats: group the clean pitcher table by team, filter `role == 'RP'`, average xBA-allowed and K%. One row per team-season.

### 10.1 `src/mlbhit/features/bullpen.py`

```python
from __future__ import annotations
import pandas as pd
from ..io import clean_path

def build_bullpen_features(seasons: list[int]) -> pd.DataFrame:
    pit = pd.read_parquet(clean_path("pitcher_season_stats.parquet"))
    pit = pit[pit["role"] == "RP"].copy()
    pit = pit[pit["season"].isin(seasons)]
    agg = pit.groupby(["team","season"]).agg(
        pen_xba_allowed=("xba_allowed","mean"),
        pen_k_pct=("k_pct_allowed","mean"),
    ).reset_index()
    agg.to_parquet(clean_path("bullpen_features.parquet"), index=False)
    return agg
```

Then in `build_features.py`, replace the placeholder line with:

```python
pen = pd.read_parquet(clean_path("bullpen_features.parquet"))
# opponent's bullpen
box = box.merge(pen.rename(columns={"team":"opponent"}),
                on=["opponent","season"], how="left")
```

---

## 11. Training the Model

### 11.1 `src/mlbhit/model/train.py`

```python
from __future__ import annotations
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from ..io import modeling_path
from ..config import SETTINGS

FEATURES = [
    # batter
    "bat_xba_season","bat_k_pct","bat_contact_pct","bat_hard_hit_pct",
    "bat_sweet_spot_pct","bat_line_drive_pct","bat_solid_contact_pct",
    # pitcher
    "sp_xba_allowed","sp_k_pct","sp_hard_hit_allowed","sp_sweet_spot_allowed",
    # bullpen
    "pen_xba_allowed","pen_k_pct",
    # context
    "exp_pa","batting_order","platoon_advantage","home_away_is_home",
    "park_factor","singles_park_factor",
    # derived
    "xba_diff","exposure_wtd_opp_xba",
    # control
    "pitcher_low_sample",
]

# Monotonic constraints: +1 up is good for the batter, -1 down
MONO = {
    "bat_xba_season":        +1,
    "bat_contact_pct":       +1,
    "bat_hard_hit_pct":      +1,
    "bat_sweet_spot_pct":    +1,
    "bat_line_drive_pct":    +1,
    "bat_solid_contact_pct": +1,
    "bat_k_pct":             -1,
    "sp_xba_allowed":        +1,
    "sp_hard_hit_allowed":   +1,
    "sp_sweet_spot_allowed": +1,
    "sp_k_pct":              -1,
    "pen_xba_allowed":       +1,
    "pen_k_pct":             -1,
    "exp_pa":                +1,
    "platoon_advantage":     +1,
    "xba_diff":              +1,
    "park_factor":           +1,
    "singles_park_factor":   +1,
}

def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df["home_away_is_home"] = (df["home_away"] == "H").astype(int)
    df["batting_order"] = df["batting_order"].fillna(5).astype(int)
    for c in FEATURES:
        if c not in df.columns:
            df[c] = 0.0
    X = df[FEATURES].fillna(df[FEATURES].median(numeric_only=True))
    y = df["got_hit"].astype(int)
    return X, y

def monotone_tuple() -> tuple:
    return tuple(MONO.get(f, 0) for f in FEATURES)

def train(df: pd.DataFrame, val_frac: float = 0.15, model_name: str = "xgb_v1") -> dict:
    df = df.sort_values("date").reset_index(drop=True)
    X, y = prepare(df)
    split = int(len(df) * (1 - val_frac))
    X_tr, y_tr = X.iloc[:split], y.iloc[:split]
    X_val, y_val = X.iloc[split:], y.iloc[split:]

    base = XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.04,
        subsample=0.85, colsample_bytree=0.85,
        reg_lambda=1.0,
        eval_metric="logloss", tree_method="hist", n_jobs=-1,
        monotone_constraints=monotone_tuple(),
    )
    calibrated = CalibratedClassifierCV(base, method="isotonic", cv=3)
    calibrated.fit(X_tr, y_tr)

    p_val = calibrated.predict_proba(X_val)[:, 1]
    p_base_const = np.full(len(y_val), y_tr.mean())

    metrics = {
        "log_loss":     float(log_loss(y_val, p_val)),
        "log_loss_const": float(log_loss(y_val, p_base_const)),
        "brier":        float(brier_score_loss(y_val, p_val)),
        "roc_auc":      float(roc_auc_score(y_val, p_val)),
        "n_train":      int(len(X_tr)),
        "n_val":        int(len(X_val)),
        "hit_rate_train": float(y_tr.mean()),
    }

    out_model = Path(SETTINGS["paths"]["models_dir"]) / f"{model_name}.joblib"
    out_meta  = Path(SETTINGS["paths"]["models_dir"]) / f"{model_name}.json"
    joblib.dump({"model": calibrated, "features": FEATURES}, out_model)
    with open(out_meta, "w") as f:
        json.dump({"features": FEATURES, "metrics": metrics, "monotone": MONO}, f, indent=2)
    return metrics

if __name__ == "__main__":
    df = pd.read_parquet(modeling_path("player_game_features.parquet"))
    m = train(df)
    print(json.dumps(m, indent=2))
```

Run:

```bash
python -m mlbhit.model.train
```

**What to expect on first run:**
- Log loss around 0.605–0.625 on validation
- `log_loss_const` around 0.645
- ROC AUC around 0.60–0.64 (this is low by general ML standards but reasonable here)

If log loss is **below 0.58**, you almost certainly have leakage. Check that:
- Training features don't include the current-game result
- Rolling features are lagged by 1 day
- You didn't accidentally fit on validation rows

### 11.2 `src/mlbhit/model/predict.py`

```python
from __future__ import annotations
import joblib
import pandas as pd
from pathlib import Path
from ..config import SETTINGS
from .train import prepare

def load_model(name: str = "xgb_v1"):
    p = Path(SETTINGS["paths"]["models_dir"]) / f"{name}.joblib"
    bundle = joblib.load(p)
    return bundle["model"], bundle["features"]

def predict(df: pd.DataFrame, name: str = "xgb_v1") -> pd.Series:
    model, feats = load_model(name)
    X, _ = prepare(df.assign(got_hit=0))  # y isn't used
    return pd.Series(model.predict_proba(X[feats])[:, 1], index=df.index, name="p_model")
```

---

## 12. Daily Scoring

### 12.1 `src/mlbhit/pipeline/score_today.py`

```python
from __future__ import annotations
from datetime import date
import pandas as pd
from ..pipeline.fetch_schedule import fetch_schedule
from ..pipeline.fetch_lineups import fetch_lineups
from ..features.batter import build_batter_features
from ..features.pitcher import build_pitcher_features
from ..features.bullpen import build_bullpen_features
from ..features.pa import expected_pa
from ..features.park_weather import attach_park
from ..model.predict import predict
from ..io import output_path, clean_path

def score_for_date(d: date, season: int) -> pd.DataFrame:
    sched = fetch_schedule(d)
    lineups = fetch_lineups(d)
    if lineups.empty:
        print("No lineups available yet; skipping.")
        return pd.DataFrame()

    # enrich lineup rows with opponent starter
    sched_lite = sched[["game_pk","home_probable_pitcher_id","away_probable_pitcher_id"]]
    df = lineups.merge(sched_lite, on="game_pk", how="left")
    df["opp_sp_id"] = df.apply(
        lambda r: r["away_probable_pitcher_id"] if r["home_away"] == "H" else r["home_probable_pitcher_id"],
        axis=1,
    )

    # batter features
    all_bat = build_batter_features(season)
    df = df.merge(all_bat, on="mlbam_id", how="inner")  # inner = drops <50 AB batters

    # pitcher
    all_pit = build_pitcher_features(season).add_prefix("sp_")
    df = df.merge(all_pit, left_on="opp_sp_id", right_on="sp_mlbam_id", how="left")
    league = all_pit[["sp_xba_allowed","sp_k_pct","sp_hard_hit_allowed","sp_sweet_spot_allowed"]].mean()
    for c, v in league.items():
        df[c] = df[c].fillna(v)
    df["pitcher_low_sample"] = df["sp_mlbam_id"].isna().astype(int)

    # bullpen (season-level table)
    pen = pd.read_parquet(clean_path("bullpen_features.parquet"))
    df = df.merge(pen[pen["season"] == season].rename(columns={"team":"opponent"}),
                  on="opponent", how="left")

    # expected PA
    df["exp_pa"] = df.apply(lambda r: expected_pa(int(r["lineup_spot"]), r["home_away"]), axis=1)

    # park (need home team for park)
    df["home_team"] = df.apply(
        lambda r: r["team"] if r["home_away"] == "H" else r["opponent"], axis=1)
    df = attach_park(df)

    # platoon: requires batter handedness + sp_throws; v1 placeholder
    df["platoon_advantage"] = 0

    # derived
    df["xba_diff"] = df["bat_xba_season"] - df["sp_xba_allowed"]
    df["exposure_wtd_opp_xba"] = 0.7 * df["sp_xba_allowed"] + 0.3 * df["pen_xba_allowed"]
    df["date"] = d.isoformat()
    df["batting_order"] = df["lineup_spot"]

    # predict
    df["p_model"] = predict(df).values

    # output
    cols_out = ["date","game_pk","player_id","player_name","team","opponent",
                "home_away","lineup_spot","exp_pa","p_model"]
    out = df[cols_out].sort_values("p_model", ascending=False)
    out.to_parquet(output_path("predictions", f"{d.isoformat()}.parquet"), index=False)
    return out

if __name__ == "__main__":
    from datetime import date
    out = score_for_date(date.today(), season=date.today().year)
    print(out.head(20).to_string(index=False))
```

### 12.2 `src/mlbhit/pipeline/recommend.py`

```python
from __future__ import annotations
import pandas as pd
from ..utils.odds_math import prob_to_american, ev_per_unit
from ..config import SETTINGS
from ..io import output_path

EDGE_MIN = SETTINGS["edge_threshold"]

def recommend(predictions: pd.DataFrame, prop_prices: pd.DataFrame | None = None) -> pd.DataFrame:
    preds = predictions.copy()
    preds["fair_american"] = preds["p_model"].apply(prob_to_american)
    if prop_prices is None or prop_prices.empty:
        # v1 without prop prices: just show the top N
        preds["edge"] = None
        return preds.head(25)

    # expected: prop_prices has columns [date, player_id, book, over_price]
    m = preds.merge(prop_prices, on=["date","player_id"], how="inner")
    m["edge"] = m.apply(lambda r: ev_per_unit(r["p_model"], int(r["over_price"])), axis=1)
    m = m[m["edge"] >= EDGE_MIN].sort_values("edge", ascending=False)
    return m

if __name__ == "__main__":
    from datetime import date
    import pandas as pd
    d = date.today().isoformat()
    preds = pd.read_parquet(output_path("predictions", f"{d}.parquet"))
    recs = recommend(preds)
    print(recs.head(15).to_string(index=False))
    recs.to_csv(output_path("recommendations", f"{d}.csv"), index=False)
```

### 12.3 `run_daily.py`

```python
#!/usr/bin/env python
from datetime import date
from mlbhit.pipeline.fetch_schedule import fetch_schedule
from mlbhit.pipeline.fetch_lineups import fetch_lineups
from mlbhit.pipeline.score_today import score_for_date
from mlbhit.pipeline.recommend import recommend
from mlbhit.io import output_path
import pandas as pd

def main():
    today = date.today()
    fetch_schedule(today)
    fetch_lineups(today)
    preds = score_for_date(today, season=today.year)
    if preds.empty:
        print("No predictions produced — maybe lineups not out yet.")
        return
    recs = recommend(preds)
    out_csv = output_path("recommendations", f"{today.isoformat()}.csv")
    recs.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(recs)} rows")
    print(recs.head(15).to_string(index=False))

if __name__ == "__main__":
    main()
```

Run:

```bash
python run_daily.py
```

---

## 13. Evaluation Notebook

Create `notebooks/eval_v1.ipynb`. The key cells:

```python
# Cell 1 - imports
import pandas as pd, numpy as np, joblib, matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from mlbhit.model.train import prepare, FEATURES
from mlbhit.io import modeling_path

bundle = joblib.load("models/xgb_v1.joblib")
model = bundle["model"]

df = pd.read_parquet(modeling_path("player_game_features.parquet")).sort_values("date")
X, y = prepare(df)
split = int(len(df) * 0.85)
X_val, y_val = X.iloc[split:], y.iloc[split:]
p = model.predict_proba(X_val)[:, 1]

# Cell 2 - metrics
print("log loss:", log_loss(y_val, p))
print("brier   :", brier_score_loss(y_val, p))
print("auc     :", roc_auc_score(y_val, p))
print("base   :", log_loss(y_val, np.full(len(y_val), y_val.mean())))

# Cell 3 - calibration plot
frac_pos, mean_pred = calibration_curve(y_val, p, n_bins=15, strategy="quantile")
plt.figure(figsize=(6,6))
plt.plot(mean_pred, frac_pos, marker="o")
plt.plot([0,1],[0,1], "k--", alpha=0.5)
plt.xlabel("Predicted P(hit)"); plt.ylabel("Observed hit rate")
plt.title("Calibration")
plt.grid(alpha=0.3); plt.show()

# Cell 4 - baseline comparisons
ba = df.iloc[split:]["bat_xba_season"]
exp_pa = df.iloc[split:]["exp_pa"]
p_baseline_xba = 1 - (1 - ba) ** exp_pa
print("xBA-PA baseline log loss:", log_loss(y_val, p_baseline_xba.clip(0.01, 0.99)))
```

What you want to see on the calibration plot: dots hugging the 45° line. A droop above 0.7 means overconfidence on high-probability bets (dangerous). A lift below 0.3 means too cautious (usually harmless for betting).

---

## 14. Seven-Day Execution Plan (concrete)

| Day | What to run | Expected output | Time |
|---|---|---|---|
| **1** | `brew`/`pyenv`/`venv`/`pip install -r requirements.txt`/`pip install -e .`; fill `.env`; `init_db`. | `python -c "from mlbhit.config import SETTINGS; print(SETTINGS)"` prints paths. DB file exists. | 2 hrs |
| **2** | `python -m mlbhit.pipeline.build_player_map` then `python -m mlbhit.pipeline.fetch_batting_stats` then `python -m mlbhit.pipeline.fetch_pitching_stats`. | 3 parquets in `data/clean/`. Confirm filters worked (`AB >= 50`). | 1 hr |
| **3** | `python -m mlbhit.pipeline.fetch_statcast` (slow). Build bullpen features. Pull schedules for 2024. | Statcast LA features in `data/clean/`; `bullpen_features.parquet`. | 1 hr runtime + 30 min coding |
| **4** | `python -m mlbhit.pipeline.fetch_boxscores` for 2023+2024 (slow). | `boxscores_2023.parquet`, `boxscores_2024.parquet`. | 1–2 hrs runtime |
| **5** | `python -m mlbhit.pipeline.build_features`. Iterate 3–5 times as errors surface. | `data/modeling/player_game_features.parquet`, ~180k rows. Print `df.describe()`. | 2–4 hrs |
| **6** | `python -m mlbhit.model.train`. Read the metrics JSON. Run the eval notebook. | `models/xgb_v1.joblib`. Log loss < 0.620 on validation. Calibration plot. | 2 hrs |
| **7** | `python run_daily.py` for today. Debug join issues. Top 25 predictions print. | `data/output/predictions/YYYY-MM-DD.parquet`, `recommendations/YYYY-MM-DD.csv`. | 2–4 hrs |

---

## 15. What to Bolt On Next (after v1 works)

1. **Platoon handedness features.** Enrich `players.parquet` with `bat_side` / `throws`. Add `platoon_advantage = int(bat_side != sp_throws)`.
2. **Batter-vs-hand splits.** Pull Statcast `xba` split by opposing pitcher hand; feature `bat_xba_vs_hand`.
3. **Rolling 14/30-day batter form.** Recompute from Statcast pitches with lag.
4. **Weather features in daily scoring.** Currently weather is pulled but not scored; add to `score_today.py`.
5. **Bullpen rolling form.** Recompute bullpen xBA over last 14 days using Statcast.
6. **Prop price ingestion.** The Odds API paid tier → `fetch_prop_odds.py` → wire to `recommend.py`.
7. **Closing line tracking.** Every morning, re-pull yesterday's closing odds; compute CLV.
8. **Approach B per-PA model.** Train on pitch-level data; combine with expected PA.
9. **Streamlit dashboard.** Today's picks, equity curve, calibration over last 30 days.
10. **launchd scheduling.** Two runs per day, morning + pre-game.

---

## 16. Things That Will Break (be ready)

| Symptom | Likely cause | Fix |
|---|---|---|
| `pybaseball` ImportError on some function | Version drift | `pip install -U pybaseball` |
| `pybaseball.pitching_stats` missing columns | FanGraphs layout change | Inspect `df.columns`; update `KEEP` list |
| Empty schedule for today | Early morning or off-season | Check date; check `statsapi.schedule(date=)` manually |
| Lineups DataFrame empty | Lineups not posted yet | Re-run closer to first pitch |
| Model log loss suspiciously low (<0.58) | Leakage | Check feature lagging; check for target-derived features |
| Predictions all cluster around 0.65 | Model learned only the prior | Check `predict` is using the right feature order; inspect `model.feature_importances_` |
| Many batters dropped from daily predictions | April effect, <50 AB | Expected early season; consider lowering filter to 30 AB in April |
| Opposing pitcher NaN on all rows | Probable pitcher not listed | Fallback to team SP rotation; log `pitcher_low_sample=1` |
| Weather API 429 | Rate limit | Cache; add sleep between calls |
| Odds API 422 | No MLB games that day | Log and skip |

---

## 17. Cursor-Specific Tips

- `Cmd+K` with a file selected: ask the AI to refactor or debug inline.
- `Cmd+L` for chat with the repo as context. Ask things like: *"explain why `build_features.py` might produce duplicate rows"* — it's much more useful with your actual code visible.
- Add a `.cursorignore` similar to `.gitignore` so the AI doesn't try to index `data/raw/statcast/pitches_2024.parquet` (which is huge).
- Keep `.env` out of AI context: add `.env` to `.cursorignore`.

`.cursorignore`:

```
.venv/
data/
db/
models/
logs/
.env
```

---

## 18. Opinionated Final Checklist

- [ ] Set `MIN_BATTER_AB = 50` in `config/settings.yaml` ✔ already done
- [ ] xBA is the anchor feature for hitters, not BA
- [ ] Launch-angle signal fed via `sweet_spot_pct`, `line_drive_pct`, `solid_contact_pct`
- [ ] Monotonic constraints ON in XGBoost
- [ ] Isotonic calibration wrapped around the classifier
- [ ] Time-based train/val split, no shuffling
- [ ] Report log loss vs the constant baseline every training run
- [ ] Log every prediction to SQLite for future CLV/backtesting
- [ ] Flat-stake until 500+ graded bets
- [ ] De-vig both sides of every market before computing edge

Ship v1, then iterate.
