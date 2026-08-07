# Phase 3: Inference

[Previous: Training](training.md) | [Back to the project overview](../README.md)

This phase applies existing CNN and RNN checkpoints to prepared model-input workbooks and writes harvest-window prediction CSV files.

Inference is intended to run on an NVIDIA GPU. Before starting, complete the shared [environment setup](../README.md#create-the-conda-environment) and create the target year's [model-input workbooks](data-download.md#create-the-model-input-workbooks).

## Files Used in This Phase

| Task | File | Main output |
|---|---|---|
| Run CNN inference | `doy_prediction/predict_tile_cnn.py` | Prediction CSV |
| Run RNN inference | `doy_prediction/predict_tile_rnn.py` | Prediction CSV |
| Combine CNN and RNN inference | `doy_prediction/predict_tile_hybrid_infer.py` | Late-fusion prediction CSV |
| Run the complete tested 2025 workflow | `scripts/run_ar_current_2025.sh` | Downloaded data, workbooks, and archived-model predictions |

## Run Inference with Existing CNN and RNN Checkpoints

The archived model can be used without retraining after Git LFS has downloaded the checkpoints and model-input workbooks have been prepared. The late-fusion implementation is `doy_prediction/predict_tile_hybrid_infer.py`.

```bash
export REPO_ROOT="$PWD"
export MODEL_ROOT="$REPO_ROOT/outputs_prediction_DOY/models/all_crops_doy_window_sweep_2022_train_2023_test_20260628_144732"

python -m doy_prediction.predict_tile_hybrid_infer \
  --inputs outputs/2023_AR \
  --cnn-checkpoints "$MODEL_ROOT/cnn/1year" \
  --rnn-checkpoints "$MODEL_ROOT/rnn/1year" \
  --feature-set all_indices \
  --out-csv predictions_2023_1year_all_indices.csv \
  --device cuda \
  --min-points 2
```

For a `6mo` or `9mo` checkpoint, pass its matching window file:

```bash
--crop-windows-yaml configs/Arkansas/doy_input_first_6mo.yaml
```

or:

```bash
--crop-windows-yaml configs/Arkansas/doy_input_first_9mo.yaml
```

Do not combine a checkpoint with a different temporal-window configuration.

## Reproduce the Tested 2025 Workflow

The 2025 runner performs the complete workflow used for the archived inference results:

1. Download 2025 Sentinel-2 data.
2. Prepare the 2025 CDL for every tile.
3. Validate every raster grid.
4. Build workbook-only inputs.
5. Run all three archived windows and both feature sets for 2025.

Use the activated Conda environment for both data processing and model inference:

```bash
export DOWNLOAD_PYTHON="$CONDA_PREFIX/bin/python"
export MODEL_PYTHON="$CONDA_PREFIX/bin/python"
export DATASET_BASE=/absolute/path/to/AR_sentinel2
export REFERENCE_ROOT="$DATASET_BASE/2023_AR"
export DEVICE=cuda

bash scripts/run_ar_current_2025.sh
```

Outputs are written to:

```text
$DATASET_BASE/2025_AR
outputs/2025_AR
outputs_prediction_DOY/inference_2022train_current_2025
```

The archived inference directory is stored in Git and contains the tested 2025 predictions for the `6mo`, `9mo`, and `1year` windows with both `all_indices` and `ndvi_only` features.
