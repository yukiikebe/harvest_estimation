#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ACTIVE_PYTHON="${CONDA_PREFIX:+${CONDA_PREFIX}/bin/python}"
DOWNLOAD_PYTHON="${DOWNLOAD_PYTHON:-${ACTIVE_PYTHON}}"
MODEL_PYTHON="${MODEL_PYTHON:-${ACTIVE_PYTHON}}"
DATASET_BASE="${DATASET_BASE:?Set DATASET_BASE to the AR_sentinel2 directory}"
REFERENCE_ROOT="${REFERENCE_ROOT:-${DATASET_BASE}/2023_AR}"
MODEL_ROOT="${MODEL_ROOT:-${ROOT_DIR}/outputs_prediction_DOY/models/all_crops_doy_window_sweep_2022_train_2023_test_20260628_144732}"
DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-4}"
PREP_WORKERS="${PREP_WORKERS:-8}"
DEVICE="${DEVICE:-cuda}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/logs/ar_current_2025_${RUN_STAMP}}"
INFER_ROOT="${INFER_ROOT:-${ROOT_DIR}/outputs_prediction_DOY/inference_2022train_current_2025}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-deepsat-current}"
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
mkdir -p "${LOG_ROOT}" "${INFER_ROOT}" "${MPLCONFIGDIR}"

if [[ -z "${DOWNLOAD_PYTHON}" || ! -x "${DOWNLOAD_PYTHON}" ]]; then
  echo "Download Python not found. Activate Conda or set DOWNLOAD_PYTHON." >&2
  exit 1
fi
if [[ -z "${MODEL_PYTHON}" || ! -x "${MODEL_PYTHON}" ]]; then
  echo "Model Python not found. Activate Conda or set MODEL_PYTHON." >&2
  exit 1
fi

run_phase() {
  local phase="$1"
  shift
  echo "[$(date -Is)] START ${phase}" | tee -a "${LOG_ROOT}/pipeline.log"
  "$@" 2>&1 | tee "${LOG_ROOT}/${phase}.log"
  echo "[$(date -Is)] END ${phase}" | tee -a "${LOG_ROOT}/pipeline.log"
}

run_phase download_2025 \
  "${DOWNLOAD_PYTHON}" -u scripts/download_ar_sentinel_stac.py \
    --year 2025 \
    --reference-root "${REFERENCE_ROOT}" \
    --output-root "${DATASET_BASE}/2025_AR" \
    --workers "${DOWNLOAD_WORKERS}"

run_phase prepare_cdl_2025 \
  "${DOWNLOAD_PYTHON}" -u scripts/prepare_ar_cdl_proxy.py \
    --cdl-year 2025 \
    --target-years 2025 \
    --dataset-base "${DATASET_BASE}" \
    --reference-root "${REFERENCE_ROOT}" \
    --workers "${DOWNLOAD_WORKERS}"

run_phase validate_downloads \
  "${DOWNLOAD_PYTHON}" -u scripts/validate_ar_current_data.py \
    --years 2025 \
    --dataset-base "${DATASET_BASE}" \
    --reference-root "${REFERENCE_ROOT}" \
    --out-json "${INFER_ROOT}/data_validation.json"

run_phase prepare_inputs_2025 env \
  DATASET_BASE="${DATASET_BASE}" \
  CONDA_ENV_PREFIX="$(dirname "$(dirname "${MODEL_PYTHON}")")" \
  PYTHON="${MODEL_PYTHON}" \
  WORKBOOK_ONLY=1 \
  RESET_CHECKPOINTS=1 \
  OVERWRITE_OUTPUTS=1 \
  scripts/run_prepare_doy_inputs_parallel.sh 2025 "${PREP_WORKERS}"

for window in 6mo 9mo 1year; do
  window_yaml=()
  if [[ "${window}" == "6mo" ]]; then
    window_yaml=(--crop-windows-yaml configs/Arkansas/doy_input_first_6mo.yaml)
  elif [[ "${window}" == "9mo" ]]; then
    window_yaml=(--crop-windows-yaml configs/Arkansas/doy_input_first_9mo.yaml)
  fi
  for feature_set in all_indices ndvi_only; do
    output_dir="${INFER_ROOT}/2025/${window}/${feature_set}"
    mkdir -p "${output_dir}"
    run_phase "infer_2025_${window}_${feature_set}" \
      "${MODEL_PYTHON}" -m doy_prediction.predict_tile_hybrid_infer \
        --inputs outputs/2025_AR \
        --cnn-checkpoints "${MODEL_ROOT}/cnn/${window}" \
        --rnn-checkpoints "${MODEL_ROOT}/rnn/${window}" \
        --feature-set "${feature_set}" \
        --out-csv "${output_dir}/predictions.csv" \
        --device "${DEVICE}" \
        --min-points 2 \
        "${window_yaml[@]}"
  done
done

"${MODEL_PYTHON}" - <<'PY' "${INFER_ROOT}" | tee "${LOG_ROOT}/inference_summary.log"
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = []
for path in sorted(root.glob("*/*/*/predictions.csv")):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    summary.append(
        {
            "path": str(path.relative_to(root)),
            "rows": len(rows),
            "crops": len({row["crop"] for row in rows}),
            "tiles": len({row["tile"] for row in rows}),
        }
    )
(root / "inference_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(summary, indent=2))
PY

echo "[$(date -Is)] PIPELINE COMPLETE infer_root=${INFER_ROOT}" | tee -a "${LOG_ROOT}/pipeline.log"
