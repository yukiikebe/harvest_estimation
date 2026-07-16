#!/usr/bin/env bash
set -euo pipefail

YEAR="${1:?usage: $0 YEAR [WORKERS]}"
WORKERS="${2:-8}"
RESET_CHECKPOINTS="${RESET_CHECKPOINTS:-1}"
OVERWRITE_OUTPUTS="${OVERWRITE_OUTPUTS:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-deepsatmodels_env}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/home/yikebe/.conda/envs/${CONDA_ENV_NAME}}"
PYTHON="${PYTHON:-${CONDA_ENV_PREFIX}/bin/python}"
DATASET_BASE="${DATASET_BASE:-/home/yikebe/AR_sentinel2}"
DATASET_ROOT="${DATASET_ROOT:-${DATASET_BASE}/${YEAR}_AR}"
OUTPUT_ROOT="${REPO_ROOT}/outputs/${YEAR}_AR"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/harvest_parallel_${YEAR}_${RUN_ID}}"

mkdir -p "${LOG_ROOT}"
cd "${REPO_ROOT}"

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "Dataset root not found: ${DATASET_ROOT}" >&2
  exit 1
fi

mapfile -t TILES < <(find "${DATASET_ROOT}" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort)
if [[ "${#TILES[@]}" -eq 0 ]]; then
  echo "No tiles found under: ${DATASET_ROOT}" >&2
  exit 1
fi

echo "year=${YEAR}" | tee "${LOG_ROOT}/run.log"
echo "workers=${WORKERS}" | tee -a "${LOG_ROOT}/run.log"
echo "reset_checkpoints=${RESET_CHECKPOINTS}" | tee -a "${LOG_ROOT}/run.log"
echo "overwrite_outputs=${OVERWRITE_OUTPUTS}" | tee -a "${LOG_ROOT}/run.log"
echo "tiles=${#TILES[@]}" | tee -a "${LOG_ROOT}/run.log"
echo "dataset_root=${DATASET_ROOT}" | tee -a "${LOG_ROOT}/run.log"
echo "output_root=${OUTPUT_ROOT}" | tee -a "${LOG_ROOT}/run.log"
echo "log_root=${LOG_ROOT}" | tee -a "${LOG_ROOT}/run.log"

for ((worker=0; worker<WORKERS; worker++)); do
  shard_file="${LOG_ROOT}/tiles_worker_${worker}.txt"
  : > "${shard_file}"
done

for ((i=0; i<${#TILES[@]}; i++)); do
  worker=$((i % WORKERS))
  printf '%s\n' "${TILES[$i]}" >> "${LOG_ROOT}/tiles_worker_${worker}.txt"
done

pids=()
for ((worker=0; worker<WORKERS; worker++)); do
  shard_file="${LOG_ROOT}/tiles_worker_${worker}.txt"
  worker_log="${LOG_ROOT}/worker_${worker}.log"
  mapfile -t SHARD_TILES < "${shard_file}"
  echo "worker=${worker} shard_tiles=${#SHARD_TILES[@]} log=${worker_log}" | tee -a "${LOG_ROOT}/run.log"

  (
    export MKL_THREADING_LAYER=GNU
    export MPLCONFIGDIR="/tmp/matplotlib-cache-${USER}"
    mkdir -p "${MPLCONFIGDIR}"
    args=(
      -m create_doy_prediction_input.main
      --dataset-root "${DATASET_ROOT}"
      --cdl-yaml "${REPO_ROOT}/configs/Arkansas/cdl.yaml"
      --gt-windows-yaml "${REPO_ROOT}/configs/Arkansas/gt_windows.yaml"
      --seeding-config-yaml "${REPO_ROOT}/configs/Arkansas/seeding_config.yaml"
      --output-root "${OUTPUT_ROOT}"
      --all-crops
      --summarize
      --cleanup-npy
      --log-dir "${LOG_ROOT}"
      --log-name "harvest_${YEAR}_worker_${worker}"
      --tiles "${SHARD_TILES[@]}"
    )
    if [[ "${RESET_CHECKPOINTS}" == "1" ]]; then
      args+=(--reset-checkpoints)
    fi
    if [[ "${OVERWRITE_OUTPUTS}" == "1" ]]; then
      args+=(--overwrite-outputs)
    fi
    "${PYTHON}" "${args[@]}"
  ) > "${worker_log}" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

echo "completed_at=$(date -Is) status=${status}" | tee -a "${LOG_ROOT}/run.log"
exit "${status}"
