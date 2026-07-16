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
OUTPUTS_ROOT="${OUTPUTS_ROOT:-$ROOT_DIR/outputs}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT_DIR/outputs_prediction_DOY/models/harvest_hybrid_window_sweep_2022_train_2023_test}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/harvest_hybrid_window_sweep_2022_train_2023_test}"
WANDB_DIR="${WANDB_DIR:-$ROOT_DIR/wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-DeepSatModels-harvest}"
WANDB_MODE="${WANDB_MODE:-online}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
MIN_POINTS="${MIN_POINTS:-2}"
CNN_HIDDEN_CHANNELS="${CNN_HIDDEN_CHANNELS:-32 64 128}"
RNN_HIDDEN_SIZES="${RNN_HIDDEN_SIZES:-32 64 128}"
DROPOUT="${DROPOUT:-0.2}"
RNN_TYPE="${RNN_TYPE:-gru}"

export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
export WANDB_DIR

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  if ! "$PYTHON" -c "import wandb; raise SystemExit(0 if wandb.Api().api_key else 1)" >/dev/null 2>&1; then
    echo "WANDB_API_KEY is not set and wandb credentials were not found. Run wandb login or export WANDB_API_KEY before running this script." >&2
    exit 1
  fi
fi

mkdir -p "$RESULTS_ROOT" "$LOG_DIR" "$WANDB_DIR"
STATUS_TSV="$RESULTS_ROOT/run_status.tsv"
printf "window\tfeature_set\tcrop\tstatus\texit_code\tlog_path\n" > "$STATUS_TSV"

WINDOW_NAMES=("6mo" "9mo" "1year")
WINDOW_RUN_LABELS=("hybrid_only6month_train2022" "hybrid_only9month_train2022" "hybrid_only12month_train2022")
WINDOW_YAMLS=(
  "$ROOT_DIR/configs/Arkansas/doy_input_first_6mo.yaml"
  "$ROOT_DIR/configs/Arkansas/doy_input_first_9mo.yaml"
  ""
)
FEATURE_SETS=("all_indices" "ndvi_only")
CROPS=("Corn" "Rice" "Soybeans")

read -r -a CNN_HIDDEN_CHANNELS_ARGS <<< "$CNN_HIDDEN_CHANNELS"
read -r -a RNN_HIDDEN_SIZES_ARGS <<< "$RNN_HIDDEN_SIZES"

ANY_FAILED=0

for window_index in "${!WINDOW_NAMES[@]}"; do
  WINDOW_NAME="${WINDOW_NAMES[$window_index]}"
  WINDOW_RUN_LABEL="${WINDOW_RUN_LABELS[$window_index]}"
  WINDOW_YAML="${WINDOW_YAMLS[$window_index]}"

  for FEATURE_SET in "${FEATURE_SETS[@]}"; do
    for CROP in "${CROPS[@]}"; do
      RUN_NAME="${WINDOW_NAME}_${FEATURE_SET}_${CROP}"
      WANDB_RUN_NAME="${WINDOW_RUN_LABEL}_${FEATURE_SET}_${CROP}"
      LOG_PATH="$LOG_DIR/${RUN_NAME}.log"
      SAVE_DIR="$RESULTS_ROOT/$WINDOW_NAME"
      CMD=(
        "$PYTHON" -m doy_prediction.train_tile_hybrid
        --outputs-root "$OUTPUTS_ROOT"
        --crops "$CROP"
        --feature-set "$FEATURE_SET"
        --train-years 2022
        --test-years 2023
        --save-dir "$SAVE_DIR"
        --epochs "$EPOCHS"
        --batch-size "$BATCH_SIZE"
        --learning-rate "$LEARNING_RATE"
        --weight-decay "$WEIGHT_DECAY"
        --val-fraction "$VAL_FRACTION"
        --min-points "$MIN_POINTS"
        --cnn-hidden-channels "${CNN_HIDDEN_CHANNELS_ARGS[@]}"
        --rnn-hidden-sizes "${RNN_HIDDEN_SIZES_ARGS[@]}"
        --dropout "$DROPOUT"
        --rnn-type "$RNN_TYPE"
        --device "$DEVICE"
        --wandb
        --wandb-project "$WANDB_PROJECT"
        --wandb-mode "$WANDB_MODE"
        --wandb-run-name "$WANDB_RUN_NAME"
        --wandb-tags hybrid_window_sweep train2022 test2023 "$WINDOW_NAME" "$FEATURE_SET" "$CROP"
        --log-test-every-epoch
      )
      if [[ -n "$WINDOW_YAML" ]]; then
        CMD+=(--crop-windows-yaml "$WINDOW_YAML")
      fi
      if [[ -n "${WANDB_ENTITY:-}" ]]; then
        CMD+=(--wandb-entity "$WANDB_ENTITY")
      fi
      if [[ "${BIDIRECTIONAL:-0}" == "1" ]]; then
        CMD+=(--bidirectional)
      fi

      echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $RUN_NAME"
      printf "Command: " > "$LOG_PATH"
      printf "%q " "${CMD[@]}" >> "$LOG_PATH"
      printf "\n\n" >> "$LOG_PATH"

      "${CMD[@]}" >> "$LOG_PATH" 2>&1
      EXIT_CODE=$?
      if [[ "$EXIT_CODE" -eq 0 ]]; then
        STATUS="ok"
      else
        STATUS="failed"
        ANY_FAILED=1
      fi
      printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$WINDOW_NAME" "$FEATURE_SET" "$CROP" "$STATUS" "$EXIT_CODE" "$LOG_PATH" >> "$STATUS_TSV"
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] END $RUN_NAME status=$STATUS exit_code=$EXIT_CODE"
    done
  done
done

echo "Status summary: $STATUS_TSV"
exit "$ANY_FAILED"
