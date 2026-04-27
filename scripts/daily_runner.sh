#!/bin/zsh
# Daily game-day runner. Designed to be invoked by launchd (see launchd/com.mlbhit.daily.plist)
# or by hand. Exit code is non-zero on any failure so launchd logs record it.
#
# Usage:
#   ./daily_runner.sh                  # scores today
#   ./daily_runner.sh 2026-06-15       # scores a specific date (useful for backfilling)

set -euo pipefail

# ---- config -------------------------------------------------------------
PROJECT_DIR="${HOME}/Documents/Claude/Projects/mlb hit"
VENV_DIR="${PROJECT_DIR}/.venv"
PYTHON="${VENV_DIR}/bin/python"
CURRENT_SEASON=2026
PRIOR_SEASON=2025
LOG_DIR="${PROJECT_DIR}/logs"
# ------------------------------------------------------------------------

mkdir -p "${LOG_DIR}"
DATE_ARG="${1:-$(date +%F)}"
STAMP=$(date +%Y%m%dT%H%M%S)
LOG="${LOG_DIR}/${DATE_ARG}_${STAMP}.log"

{
  echo "=================================================================="
  echo "mlbhit daily run  date=${DATE_ARG}  started=$(date -Iseconds)"
  echo "=================================================================="

  cd "${PROJECT_DIR}"

  # Activate venv by invoking the venv's python directly (no `source` needed).
  # Set working directory explicitly so relative paths in the package resolve.

  echo "[1/4] Fetching today's schedule..."
  "${PYTHON}" -m mlbhit.pipeline.fetch_schedule

  echo "[2/4] Fetching today's lineups..."
  "${PYTHON}" -m mlbhit.pipeline.fetch_lineups

  echo "[3/4] Scoring ${DATE_ARG} (season=${CURRENT_SEASON}, prior=${PRIOR_SEASON})..."
  "${PYTHON}" -m mlbhit.pipeline.score_today \
      --date "${DATE_ARG}" \
      --season "${CURRENT_SEASON}" \
      --prior-season "${PRIOR_SEASON}"

  # If a manual prop CSV exists for the date, run the EV recommender.
  PROPS_CSV="${PROJECT_DIR}/data/raw/props/${DATE_ARG}_props.csv"
  if [[ -s "${PROPS_CSV}" ]]; then
    echo "[4/4] Props CSV found; loading + generating recommendations..."
    "${PYTHON}" -m mlbhit.pipeline.fetch_prop_odds --date "${DATE_ARG}" --source csv
    "${PYTHON}" -m mlbhit.pipeline.recommend
  else
    echo "[4/4] No props CSV at ${PROPS_CSV} — skipping recommendations."
    echo "      Create one with:  ${PYTHON} -m mlbhit.pipeline.fetch_prop_odds --date ${DATE_ARG} --make-template"
  fi

  echo "mlbhit daily run  finished=$(date -Iseconds)"
} >> "${LOG}" 2>&1

# Print log location to stdout so launchd's StandardOutPath picks it up.
echo "Wrote log: ${LOG}"
