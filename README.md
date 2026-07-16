# Harvest DOY Estimation

This repository contains the Arkansas harvest day-of-year (DOY) workflow:

1. Build tile/crop time-series inputs from Sentinel-2 and CDL data.
2. Train CNN, RNN, and hybrid models to predict harvest start/end DOY.
3. Archive the 2022-train/2023-test all-crops model sweep.

## Repository Layout

```text
configs/Arkansas/              YAML configs for CDL labels, harvest windows, and DOY input windows
create_doy_prediction_input/   Sentinel-2/CDL processing pipeline that creates model inputs
doy_prediction/                DOY model training, inference, aggregation, and tests
scripts/                       Reproducible run scripts
outputs_prediction_DOY/models/ Archived trained model sweep
```

The archived model run is:

```text
outputs_prediction_DOY/models/all_crops_doy_window_sweep_2022_train_2023_test_20260628_144732
```

Model checkpoints (`*.pt`, `*.pth`) are tracked with Git LFS.

## Environment

```bash
conda env create -f deepsatmodels_env_latest.yml
conda activate deepsatmodels_env
```

If the environment already exists, set `CONDA_ENV_PREFIX` or `PYTHON` when running scripts.

## Input Data

The input-generation scripts expect yearly Sentinel-2 tile directories such as:

```text
/home/yikebe/AR_sentinel2/2022_AR/
/home/yikebe/AR_sentinel2/2023_AR/
```

Override the base path with:

```bash
export DATASET_BASE=/path/to/AR_sentinel2
```

Each yearly directory should contain tile folders such as `0_0`, `0_1`, ..., with date folders and `cdl.tif`.

## Build DOY Prediction Inputs

Run one year:

```bash
scripts/run_prepare_doy_inputs_parallel.sh 2022 8
```

Run 2022 and 2023:

```bash
scripts/run_prepare_doy_inputs_2022_2023.sh 8
```

The generated yearly model inputs are written under:

```text
outputs/2022_AR
outputs/2023_AR
```

These intermediate outputs are intentionally ignored by Git.

## Train and Evaluate DOY Models

Run the all-crops CNN/RNN/hybrid sweep:

```bash
WANDB_MODE=offline scripts/run_all_crops_doy_window_sweep.sh
```

Useful overrides:

```bash
DEVICE=cuda
EPOCHS=40
BATCH_SIZE=32
OUTPUTS_ROOT=/path/to/outputs
RESULTS_ROOT=/path/to/results
TRAIN_MODELS="cnn rnn hybrid"
FEATURE_SETS="all_indices ndvi_only"
WANDB_ENABLED=0
```

Standalone scripts are also available:

```bash
scripts/run_rnn_window_sweep.sh
scripts/run_hybrid_window_sweep.sh
```

## Direct Module Usage

Prepare one year:

```bash
python -m create_doy_prediction_input.main \
  --dataset-root /home/yikebe/AR_sentinel2/2022_AR \
  --cdl-yaml configs/Arkansas/cdl.yaml \
  --gt-windows-yaml configs/Arkansas/gt_windows.yaml \
  --seeding-config-yaml configs/Arkansas/seeding_config.yaml \
  --output-root outputs/2022_AR \
  --all-crops \
  --summarize \
  --cleanup-npy
```

Train one crop/model:

```bash
python -m doy_prediction.train_tile_hybrid \
  --outputs-root outputs \
  --crops Rice \
  --feature-set all_indices \
  --train-years 2022 \
  --test-years 2023 \
  --save-dir outputs_prediction_DOY/models/example_hybrid \
  --epochs 40 \
  --batch-size 32 \
  --device cuda
```

Run hybrid inference from trained CNN/RNN checkpoints:

```bash
python -m doy_prediction.predict_tile_hybrid_infer \
  --inputs outputs/2023_AR \
  --cnn-checkpoints outputs_prediction_DOY/models/all_crops_doy_window_sweep_2022_train_2023_test_20260628_144732/cnn/1year \
  --rnn-checkpoints outputs_prediction_DOY/models/all_crops_doy_window_sweep_2022_train_2023_test_20260628_144732/rnn/1year \
  --feature-set all_indices \
  --out-csv predictions.csv \
  --device cuda
```

## Tests

```bash
pytest doy_prediction/tests
```
