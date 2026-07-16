#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKERS="${1:-8}"

cd "${REPO_ROOT}"

run_year() {
  local year="$1"
  echo "===== $(date -Is) START ${year} workers=${WORKERS} overwrite=1 reset=1 ====="
  OVERWRITE_OUTPUTS=1 RESET_CHECKPOINTS=1 "${REPO_ROOT}/scripts/run_prepare_doy_inputs_parallel.sh" "${year}" "${WORKERS}"
  echo "===== $(date -Is) DONE ${year} ====="
}

run_year 2022
run_year 2023
