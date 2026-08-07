# Phase 2: Training

[Previous: Data download](data-download.md) | [Back to the project overview](../README.md) | [Next: Inference](inference.md)

This phase trains and evaluates CNN, RNN, and hybrid models using the per-tile workbooks produced during [data preparation](data-download.md#create-the-model-input-workbooks).

Training is intended to run on an NVIDIA GPU. The reference experiment uses 2022 for training and 2023 for held-out evaluation.

## Files Used in This Phase

| Task | File | Main output |
|---|---|---|
| Train and evaluate a CNN | `doy_prediction/train_tile_cnn.py` | CNN checkpoint, metrics, and predictions |
| Train and evaluate an RNN | `doy_prediction/train_tile_rnn.py` | RNN checkpoint, metrics, and predictions |
| Train and evaluate a hybrid model | `doy_prediction/train_tile_hybrid.py` | Hybrid checkpoint, metrics, and predictions |
| Run the full model sweep | `scripts/run_all_crops_doy_window_sweep.sh` | All model, feature-set, and temporal-window results |
| Aggregate a complete sweep | `doy_prediction/aggregate_all_crops_window_results.py` | Comparison Excel workbook |

## Train and Evaluate the Full Model Sweep

The maintained sweep trains all crops listed in `configs/Arkansas/gt_windows.yaml` for:

- CNN, RNN, and end-to-end hybrid models
- `all_indices` (NDVI, NDWI, EVI) and `ndvi_only`
- `6mo`, `9mo`, and `1year` input configurations
- 2022 training with 2023 evaluation

It also runs late-fusion hybrid inference from the trained CNN and RNN checkpoints. The complete default sweep contains hundreds of GPU training jobs. The shell wrapper calls `train_tile_cnn.py`, `train_tile_rnn.py`, `train_tile_hybrid.py`, and `predict_tile_hybrid_infer.py`.

Start the full run:

```bash
export RUN_LABEL=all_crops_doy_window_sweep_2022_train_2023_test
export WANDB_ENABLED=0
export DEVICE=cuda
export EPOCHS=40
export BATCH_SIZE=32
export NUM_WORKERS=4

bash scripts/run_all_crops_doy_window_sweep.sh
```

The run directory is:

```text
outputs_prediction_DOY/models/<RUN_LABEL>/
```

Its main outputs are:

```text
run_status.tsv
cnn/<WINDOW>/<FEATURE_SET>/<CROP>/best_model.pt
cnn/<WINDOW>/<FEATURE_SET>/<CROP>/metrics.json
rnn/<WINDOW>/<FEATURE_SET>/<CROP>/best_model.pt
rnn/<WINDOW>/<FEATURE_SET>/<CROP>/metrics.json
hybrid/<WINDOW>/<FEATURE_SET>/<CROP>/best_model.pt
hybrid/<WINDOW>/<FEATURE_SET>/<CROP>/metrics.json
hybrid_infer/<WINDOW>/<FEATURE_SET>/predictions.csv
```

The runner skips an existing model when both `metrics.json` and `best_model.pt` are present. Set `RESUME_EXISTING=0` only when a deliberate rerun is required.

Weights & Biases logging is optional. The example disables it so a first run does not require an account or API key. If W&B is installed and configured, set `WANDB_ENABLED=1` and choose `WANDB_MODE=online` or `offline`.

## Temporal-Window Configuration

The `6mo` and `9mo` runs use:

```text
configs/Arkansas/doy_input_first_6mo.yaml
configs/Arkansas/doy_input_first_9mo.yaml
```

A crop listed in one of these files is restricted to that date range. A crop that is not listed uses all available dates. The `1year` run always uses all available dates.

## Read Evaluation Results

Training writes evaluation results into each crop's `metrics.json`.

| Metric | Meaning | Better |
|---|---|---|
| `mae_start` | Mean absolute error of harvest start DOY, in days | Lower |
| `mae_end` | Mean absolute error of harvest end DOY, in days | Lower |
| `mae_mean` | Mean of start and end MAE | Lower |
| `mae_window` | MAE of predicted harvest-window length, in days | Lower |
| `iou_mean` | Mean intersection-over-union of predicted and target windows | Higher |
| `ordered_pct` | Fraction with predicted start DOY no later than end DOY | Higher |

## Aggregate Evaluation Results

Create one comparison workbook after the sweep completes:

```bash
export REPO_ROOT="$PWD"
export RESULTS_ROOT="$REPO_ROOT/outputs_prediction_DOY/models/$RUN_LABEL"

python -m doy_prediction.aggregate_all_crops_window_results \
  --results-root "$RESULTS_ROOT" \
  --out-xlsx "$RESULTS_ROOT/all_crops_doy_window_summary.xlsx" \
  --train-year 2022 \
  --test-year 2023
```

Continue with [Phase 3: Inference](inference.md) to apply the resulting checkpoints to prepared yearly data.
