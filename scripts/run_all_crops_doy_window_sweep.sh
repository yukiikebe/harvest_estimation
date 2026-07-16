#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-deepsatmodels_env}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/home/yikebe/.conda/envs/$CONDA_ENV_NAME}"
if [[ ! -x "$CONDA_ENV_PREFIX/bin/python" ]]; then
  echo "Conda environment python not found: $CONDA_ENV_PREFIX/bin/python" >&2
  exit 1
fi
export CONDA_DEFAULT_ENV="$CONDA_ENV_NAME"
export CONDA_PREFIX="$CONDA_ENV_PREFIX"
export PATH="$CONDA_ENV_PREFIX/bin:$PATH"
PYTHON="${PYTHON:-$CONDA_ENV_PREFIX/bin/python}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_LABEL="${RUN_LABEL:-all_crops_doy_window_sweep_2022_train_2023_test_${RUN_STAMP}}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-$ROOT_DIR/outputs}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT_DIR/outputs_prediction_DOY/models/$RUN_LABEL}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/$RUN_LABEL}"
WANDB_DIR="${WANDB_DIR:-$ROOT_DIR/wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-DeepSatModels-harvest-all-crops}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENABLED="${WANDB_ENABLED:-1}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
MIN_POINTS="${MIN_POINTS:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CNN_HIDDEN_CHANNELS="${CNN_HIDDEN_CHANNELS:-32 64 128}"
RNN_HIDDEN_SIZES="${RNN_HIDDEN_SIZES:-32 64 128}"
RNN_HIDDEN_SIZE="${RNN_HIDDEN_SIZE:-64}"
RNN_NUM_LAYERS="${RNN_NUM_LAYERS:-2}"
DROPOUT="${DROPOUT:-0.2}"
RNN_TYPE="${RNN_TYPE:-gru}"
CROPS_YAML="${CROPS_YAML:-$ROOT_DIR/configs/Arkansas/gt_windows.yaml}"
TRAIN_MODELS="${TRAIN_MODELS:-cnn rnn hybrid}"
RUN_HYBRID_INFER="${RUN_HYBRID_INFER:-1}"
FEATURE_SETS="${FEATURE_SETS:-all_indices ndvi_only}"
RESUME_EXISTING="${RESUME_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
export WANDB_DIR

if [[ ! -f "$CROPS_YAML" ]]; then
  echo "Crop YAML not found: $CROPS_YAML" >&2
  exit 1
fi

if [[ "$WANDB_ENABLED" == "1" && "$WANDB_MODE" == "online" ]]; then
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    if ! "$PYTHON" -c "import wandb; raise SystemExit(0 if wandb.Api().api_key else 1)" >/dev/null 2>&1; then
      echo "WANDB_API_KEY is not set and wandb credentials were not found. Run wandb login, set WANDB_MODE=offline, or set WANDB_ENABLED=0." >&2
      exit 1
    fi
  fi
fi

readarray -t CROPS < <("$PYTHON" - "$CROPS_YAML" <<'PY'
import sys
from pathlib import Path
import yaml

path = Path(sys.argv[1])
data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
if not isinstance(data, dict):
    raise SystemExit(f"Expected mapping in {path}")
for crop in data:
    print(str(crop))
PY
)

if [[ "${#CROPS[@]}" -eq 0 ]]; then
  echo "No crops found in $CROPS_YAML" >&2
  exit 1
fi

safe_name() {
  "$PYTHON" - "$1" <<'PY'
import re
import sys

text = sys.argv[1].replace("/", "_").strip()
text = re.sub(r"\s+", "_", text)
text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
print(text.strip("_") or "crop")
PY
}

declare -A CROP_DIR_SEEN=()
for CROP in "${CROPS[@]}"; do
  CROP_DIR="$(safe_name "$CROP")"
  if [[ -n "${CROP_DIR_SEEN[$CROP_DIR]:-}" ]]; then
    echo "Crop output name collision: '$CROP' and '${CROP_DIR_SEEN[$CROP_DIR]}' both map to '$CROP_DIR'" >&2
    exit 1
  fi
  CROP_DIR_SEEN[$CROP_DIR]="$CROP"
done

mkdir -p "$RESULTS_ROOT" "$LOG_DIR" "$WANDB_DIR"
STATUS_TSV="$RESULTS_ROOT/run_status.tsv"
printf "phase\twindow\tfeature_set\tcrop\tstatus\texit_code\tlog_path\toutput_path\n" > "$STATUS_TSV"

WINDOW_NAMES=("6mo" "9mo" "1year")
WINDOW_RUN_LABELS=("only6month_train2022" "only9month_train2022" "only12month_train2022")
WINDOW_YAMLS=(
  "$ROOT_DIR/configs/Arkansas/doy_input_first_6mo.yaml"
  "$ROOT_DIR/configs/Arkansas/doy_input_first_9mo.yaml"
  ""
)

read -r -a FEATURE_SET_ARGS <<< "$FEATURE_SETS"
read -r -a TRAIN_MODEL_ARGS <<< "$TRAIN_MODELS"
read -r -a CNN_HIDDEN_CHANNELS_ARGS <<< "$CNN_HIDDEN_CHANNELS"
read -r -a RNN_HIDDEN_SIZES_ARGS <<< "$RNN_HIDDEN_SIZES"

contains_train_model() {
  local needle="$1"
  local item
  for item in "${TRAIN_MODEL_ARGS[@]}"; do
    if [[ "$item" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

append_common_train_args() {
  local -n cmd_ref=$1
  local crop="$2"
  local feature_set="$3"
  local save_dir="$4"
  local window_yaml="$5"
  local wandb_run_name="$6"
  local wandb_tag_model="$7"
  local window_name="$8"

  cmd_ref+=(
    --outputs-root "$OUTPUTS_ROOT"
    --crops "$crop"
    --feature-set "$feature_set"
    --train-years 2022
    --test-years 2023
    --save-dir "$save_dir"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --learning-rate "$LEARNING_RATE"
    --weight-decay "$WEIGHT_DECAY"
    --val-fraction "$VAL_FRACTION"
    --min-points "$MIN_POINTS"
    --num-workers "$NUM_WORKERS"
    --device "$DEVICE"
    --log-test-every-epoch
  )
  if [[ -n "$window_yaml" ]]; then
    cmd_ref+=(--crop-windows-yaml "$window_yaml")
  fi
  if [[ "$WANDB_ENABLED" == "1" ]]; then
    cmd_ref+=(
      --wandb
      --wandb-project "$WANDB_PROJECT"
      --wandb-mode "$WANDB_MODE"
      --wandb-run-name "$wandb_run_name"
      --wandb-tags all_crops_doy_window_sweep train2022 test2023 "$wandb_tag_model" "$window_name" "$feature_set" "$crop"
    )
    if [[ -n "${WANDB_ENTITY:-}" ]]; then
      cmd_ref+=(--wandb-entity "$WANDB_ENTITY")
    fi
  fi
}

run_command() {
  local phase="$1"
  local window_name="$2"
  local feature_set="$3"
  local crop="$4"
  local log_path="$5"
  local output_path="$6"
  shift 6
  local cmd=("$@")

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $phase $window_name $feature_set $crop"
  printf "Command: " > "$log_path"
  printf "%q " "${cmd[@]}" >> "$log_path"
  printf "\n\n" >> "$log_path"

  if [[ "$DRY_RUN" == "1" ]]; then
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$phase" "$window_name" "$feature_set" "$crop" "dry_run" "0" "$log_path" "$output_path" >> "$STATUS_TSV"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] DRY_RUN $phase $window_name $feature_set $crop"
    return 0
  fi

  "${cmd[@]}" >> "$log_path" 2>&1
  local exit_code=$?
  local status="ok"
  if [[ "$exit_code" -ne 0 ]]; then
    status="failed"
    ANY_FAILED=1
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$phase" "$window_name" "$feature_set" "$crop" "$status" "$exit_code" "$log_path" "$output_path" >> "$STATUS_TSV"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] END $phase $window_name $feature_set $crop status=$status exit_code=$exit_code"
}

ANY_FAILED=0

echo "run_label=$RUN_LABEL"
echo "results_root=$RESULTS_ROOT"
echo "log_dir=$LOG_DIR"
echo "crops=${#CROPS[@]}"
echo "feature_sets=${FEATURE_SET_ARGS[*]}"
echo "train_models=${TRAIN_MODEL_ARGS[*]}"
echo "run_hybrid_infer=$RUN_HYBRID_INFER"
echo "status_tsv=$STATUS_TSV"

for window_index in "${!WINDOW_NAMES[@]}"; do
  WINDOW_NAME="${WINDOW_NAMES[$window_index]}"
  WINDOW_RUN_LABEL="${WINDOW_RUN_LABELS[$window_index]}"
  WINDOW_YAML="${WINDOW_YAMLS[$window_index]}"

  for FEATURE_SET in "${FEATURE_SET_ARGS[@]}"; do
    for CROP in "${CROPS[@]}"; do
      CROP_SAFE="$(safe_name "$CROP")"

      if contains_train_model "cnn"; then
        SAVE_DIR="$RESULTS_ROOT/cnn/$WINDOW_NAME"
        OUTPUT_PATH="$SAVE_DIR/$FEATURE_SET/${CROP//\//_}/metrics.json"
        LOG_PATH="$LOG_DIR/cnn_${WINDOW_NAME}_${FEATURE_SET}_${CROP_SAFE}.log"
        if [[ "$RESUME_EXISTING" == "1" && -s "$OUTPUT_PATH" && -s "$SAVE_DIR/$FEATURE_SET/${CROP//\//_}/best_model.pt" ]]; then
          printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "cnn" "$WINDOW_NAME" "$FEATURE_SET" "$CROP" "skipped_existing" "0" "$LOG_PATH" "$OUTPUT_PATH" >> "$STATUS_TSV"
        else
          CMD=("$PYTHON" -m doy_prediction.train_tile_cnn)
          append_common_train_args CMD "$CROP" "$FEATURE_SET" "$SAVE_DIR" "$WINDOW_YAML" "${RUN_LABEL}_cnn_${WINDOW_RUN_LABEL}_${FEATURE_SET}_${CROP_SAFE}" "cnn" "$WINDOW_NAME"
          run_command "cnn" "$WINDOW_NAME" "$FEATURE_SET" "$CROP" "$LOG_PATH" "$OUTPUT_PATH" "${CMD[@]}"
        fi
      fi

      if contains_train_model "rnn"; then
        SAVE_DIR="$RESULTS_ROOT/rnn/$WINDOW_NAME"
        OUTPUT_PATH="$SAVE_DIR/$FEATURE_SET/${CROP//\//_}/metrics.json"
        LOG_PATH="$LOG_DIR/rnn_${WINDOW_NAME}_${FEATURE_SET}_${CROP_SAFE}.log"
        if [[ "$RESUME_EXISTING" == "1" && -s "$OUTPUT_PATH" && -s "$SAVE_DIR/$FEATURE_SET/${CROP//\//_}/best_model.pt" ]]; then
          printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "rnn" "$WINDOW_NAME" "$FEATURE_SET" "$CROP" "skipped_existing" "0" "$LOG_PATH" "$OUTPUT_PATH" >> "$STATUS_TSV"
        else
          CMD=(
            "$PYTHON" -m doy_prediction.train_tile_rnn
          )
          append_common_train_args CMD "$CROP" "$FEATURE_SET" "$SAVE_DIR" "$WINDOW_YAML" "${RUN_LABEL}_rnn_${WINDOW_RUN_LABEL}_${FEATURE_SET}_${CROP_SAFE}" "rnn" "$WINDOW_NAME"
          CMD+=(
            --hidden-size "$RNN_HIDDEN_SIZE"
            --num-layers "$RNN_NUM_LAYERS"
            --hidden-sizes "${RNN_HIDDEN_SIZES_ARGS[@]}"
            --dropout "$DROPOUT"
            --rnn-type "$RNN_TYPE"
          )
          if [[ "${BIDIRECTIONAL:-0}" == "1" ]]; then
            CMD+=(--bidirectional)
          fi
          run_command "rnn" "$WINDOW_NAME" "$FEATURE_SET" "$CROP" "$LOG_PATH" "$OUTPUT_PATH" "${CMD[@]}"
        fi
      fi

      if contains_train_model "hybrid"; then
        SAVE_DIR="$RESULTS_ROOT/hybrid/$WINDOW_NAME"
        OUTPUT_PATH="$SAVE_DIR/$FEATURE_SET/${CROP//\//_}/metrics.json"
        LOG_PATH="$LOG_DIR/hybrid_${WINDOW_NAME}_${FEATURE_SET}_${CROP_SAFE}.log"
        if [[ "$RESUME_EXISTING" == "1" && -s "$OUTPUT_PATH" && -s "$SAVE_DIR/$FEATURE_SET/${CROP//\//_}/best_model.pt" ]]; then
          printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "hybrid" "$WINDOW_NAME" "$FEATURE_SET" "$CROP" "skipped_existing" "0" "$LOG_PATH" "$OUTPUT_PATH" >> "$STATUS_TSV"
        else
          CMD=(
            "$PYTHON" -m doy_prediction.train_tile_hybrid
          )
          append_common_train_args CMD "$CROP" "$FEATURE_SET" "$SAVE_DIR" "$WINDOW_YAML" "${RUN_LABEL}_hybrid_${WINDOW_RUN_LABEL}_${FEATURE_SET}_${CROP_SAFE}" "hybrid" "$WINDOW_NAME"
          CMD+=(
            --cnn-hidden-channels "${CNN_HIDDEN_CHANNELS_ARGS[@]}"
            --rnn-hidden-size "$RNN_HIDDEN_SIZE"
            --rnn-num-layers "$RNN_NUM_LAYERS"
            --rnn-hidden-sizes "${RNN_HIDDEN_SIZES_ARGS[@]}"
            --dropout "$DROPOUT"
            --rnn-type "$RNN_TYPE"
          )
          if [[ "${BIDIRECTIONAL:-0}" == "1" ]]; then
            CMD+=(--bidirectional)
          fi
          run_command "hybrid" "$WINDOW_NAME" "$FEATURE_SET" "$CROP" "$LOG_PATH" "$OUTPUT_PATH" "${CMD[@]}"
        fi
      fi
    done

    if [[ "$RUN_HYBRID_INFER" == "1" ]]; then
      OUT_CSV="$RESULTS_ROOT/hybrid_infer/$WINDOW_NAME/$FEATURE_SET/predictions.csv"
      LOG_PATH="$LOG_DIR/hybrid_infer_${WINDOW_NAME}_${FEATURE_SET}.log"
      if [[ "$RESUME_EXISTING" == "1" && -s "$OUT_CSV" ]]; then
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "hybrid_infer" "$WINDOW_NAME" "$FEATURE_SET" "ALL" "skipped_existing" "0" "$LOG_PATH" "$OUT_CSV" >> "$STATUS_TSV"
      else
        CMD=(
          "$PYTHON" -m doy_prediction.predict_tile_hybrid_infer
          --inputs "$OUTPUTS_ROOT/2023_AR"
          --cnn-checkpoints "$RESULTS_ROOT/cnn/$WINDOW_NAME"
          --rnn-checkpoints "$RESULTS_ROOT/rnn/$WINDOW_NAME"
          --feature-set "$FEATURE_SET"
          --out-csv "$OUT_CSV"
          --device "$DEVICE"
          --min-points "$MIN_POINTS"
        )
        if [[ -n "$WINDOW_YAML" ]]; then
          CMD+=(--crop-windows-yaml "$WINDOW_YAML")
        fi
        run_command "hybrid_infer" "$WINDOW_NAME" "$FEATURE_SET" "ALL" "$LOG_PATH" "$OUT_CSV" "${CMD[@]}"
      fi
    fi
  done
done

echo "Status summary: $STATUS_TSV"
exit "$ANY_FAILED"
