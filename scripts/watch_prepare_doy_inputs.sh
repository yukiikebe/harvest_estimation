#!/usr/bin/env bash
set -euo pipefail

YEAR="${1:?usage: $0 YEAR [WORKERS] [INTERVAL_SECONDS]}"
WORKERS="${2:-8}"
INTERVAL_SECONDS="${3:-1800}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_BASE="${DATASET_BASE:-/home/yikebe/AR_sentinel2}"
OUTPUT_ROOT="${REPO_ROOT}/outputs/${YEAR}_AR"
RUNNER="${REPO_ROOT}/scripts/run_prepare_doy_inputs_parallel.sh"
SESSION="harvest${YEAR}_allcrops${WORKERS}"
WATCH_LOG="${REPO_ROOT}/logs/harvest_watch_${YEAR}_$(date +%Y%m%d_%H%M%S).log"

cd "${REPO_ROOT}"

log() {
  printf '%s %s\n' "$(date -Is)" "$*" | tee -a "${WATCH_LOG}"
}

latest_run_log() {
  find "${REPO_ROOT}/logs" -maxdepth 2 -path "*/harvest_parallel_${YEAR}_*/run.log" -printf '%T@ %p\n' \
    | sort -nr \
    | awk 'NR == 1 {print $2}'
}

completed_tiles() {
  if [[ ! -d "${OUTPUT_ROOT}" ]]; then
    echo 0
    return
  fi
  local matches
  matches="$(find "${OUTPUT_ROOT}" -maxdepth 2 -name .pipeline_checkpoint.json -exec grep -l '"tile_complete": true' {} + 2>/dev/null || true)"
  if [[ -z "${matches}" ]]; then
    echo 0
  else
    printf '%s\n' "${matches}" | wc -l
  fi
}

total_tiles() {
  find "${DATASET_BASE}/${YEAR}_AR" -maxdepth 1 -mindepth 1 -type d | wc -l
}

run_status() {
  local run_log="$1"
  if [[ -z "${run_log}" || ! -f "${run_log}" ]]; then
    echo "unknown"
    return
  fi
  if tail -20 "${run_log}" | rg -q 'status=0'; then
    echo "complete"
  elif tail -20 "${run_log}" | rg -q 'status=1'; then
    echo "failed"
  else
    echo "running_or_incomplete"
  fi
}

start_resume_session() {
  log "starting resume session ${SESSION}"
  tmux new-session -d -s "${SESSION}" "RESET_CHECKPOINTS=0 ${RUNNER} ${YEAR} ${WORKERS}"
}

log "watchdog_start year=${YEAR} workers=${WORKERS} interval_seconds=${INTERVAL_SECONDS} session=${SESSION}"

while true; do
  total="$(total_tiles | tr -d ' ')"
  completed="$(completed_tiles | tr -d ' ')"
  run_log="$(latest_run_log || true)"
  status="$(run_status "${run_log}")"

  if tmux has-session -t "${SESSION}" 2>/dev/null; then
    log "session=running completed=${completed}/${total} latest_run_log=${run_log:-none} status=${status}"
  else
    log "session=stopped completed=${completed}/${total} latest_run_log=${run_log:-none} status=${status}"
    if [[ "${status}" == "complete" || "${completed}" == "${total}" ]]; then
      log "all_done; watchdog_exit"
      exit 0
    fi
    start_resume_session
  fi

  sleep "${INTERVAL_SECONDS}"
done
