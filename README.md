# Arkansas Harvest DOY Estimation

This repository predicts crop harvest start and end day-of-year (DOY) from
Sentinel-2 vegetation-index time series. The maintained workflow is designed
for Arkansas and supports:

1. Downloading Sentinel-2 L2A imagery from the public Earth Search STAC API.
2. Downloading USDA Cropland Data Layer (CDL) masks from CropScape.
3. Building one crop-level time-series workbook per spatial tile.
4. Training CNN, RNN, and CNN/RNN hybrid models.
5. Evaluating models on a held-out year and exporting metrics and predictions.
6. Applying the archived 2022-trained models to newer years.

The reference experiment trains on 2022 and evaluates on 2023. Its checkpoints
and summary workbooks are stored under:

```text
outputs_prediction_DOY/models/all_crops_doy_window_sweep_2022_train_2023_test_20260628_144732
```

Model checkpoints (`*.pt` and `*.pth`) are managed with Git LFS.

## Workflow at a Glance

```text
Sentinel-2 L2A rasters + CDL crop masks
                    |
                    v
        NDVI / NDWI / EVI time series
                    |
                    v
 outputs/<YEAR>_AR/<TILE>/harvest_summary_all_crops.xlsx
                    |
                    v
       CNN / RNN / hybrid model training
                    |
                    v
 checkpoints + metrics.json + prediction CSV files
```

## Repository Layout

```text
configs/Arkansas/              Crop labels, harvest windows, and input windows
create_doy_prediction_input/   Converts Sentinel-2 GeoTIFFs and CDL masks into per-tile Excel time-series files used by the models
doy_prediction/                Models, training, inference, aggregation, and tests
scripts/                       Reproducible data, training, and inference commands
outputs_prediction_DOY/models/ Archived 2022-train/2023-test model sweep
deepsatmodels_env_latest.yml   Exported Conda environment
```

Large satellite data, generated model inputs, logs, and unarchived experiment
outputs are intentionally not stored in Git.

## Prerequisites

Use a Linux machine with:

- Git and Git LFS
- Conda or Miniconda
- An NVIDIA GPU and a compatible NVIDIA driver
- Enough local storage for multi-year Sentinel-2 GeoTIFF data
- Internet access to Earth Search and USDA CropScape
- An existing archived Arkansas reference grid, normally `2023_AR`

The repository does not contain the raw 2022/2023 Sentinel-2 dataset. Ask the
project maintainer for the archived `2023_AR` reference dataset before trying
to create a new aligned yearly dataset.

> The STAC downloader is not a grid bootstrapper. It reads the tile names,
> raster dimensions, CRS, and affine transforms from an existing reference
> year and reproduces that grid exactly.

## 1. Clone the Repository

Install Git LFS before retrieving the archived checkpoints:

```bash
git lfs install
git clone --branch yuki https://github.com/yukiikebe/harvest_estimation.git
cd harvest_estimation
git lfs pull
```

Run all commands below from the repository root.

## 2. Create the Conda Environment

Create the environment using the exported environment file. Supplying
`--name` overrides the machine-specific prefix stored in the exported YAML.

```bash
conda env create \
  --name deepsatmodels_env \
  --file deepsatmodels_env_latest.yml

conda activate deepsatmodels_env
```

The run scripts automatically use the active Conda environment. Training and
inference are intended to run on an NVIDIA GPU.

## 3. Configure Data Paths

Choose a location outside the repository for the raw satellite data:

```bash
export REPO_ROOT="$PWD"
export DATASET_BASE=/absolute/path/to/AR_sentinel2
export REFERENCE_ROOT="$DATASET_BASE/2023_AR"
```

The expected raw-data layout is:

```text
AR_sentinel2/
├── 2022_AR/
│   ├── 0_0/
│   │   ├── cdl.tif
│   │   ├── 2022-01-02/
│   │   │   ├── B2_2022-01-02.tif
│   │   │   ├── B4_2022-01-02.tif
│   │   │   ├── B8_2022-01-02.tif
│   │   │   ├── B11_2022-01-02.tif
│   │   │   └── SCL_2022-01-02.tif
│   │   └── ...
│   └── ...
└── 2023_AR/
    ├── 0_0/
    ├── 0_1/
    └── ...
```

Each year contains spatial tile directories. Each acquisition-date directory
contains four reflectance bands and the Sentinel-2 Scene Classification Layer
(SCL). A tile-level `cdl.tif` supplies the crop labels.

If complete 2022 and 2023 directories already exist, skip to
[Create the Excel inputs for training and inference](#5-create-the-excel-inputs-for-training-and-inference).

## 4. Download and Validate Raw Data

### 4.1 Inspect a download without writing rasters

The downloader queries public Sentinel-2 L2A COGs and maps them onto the exact
grid of `REFERENCE_ROOT`. `--end-date` is exclusive.

```bash
python scripts/download_ar_sentinel_stac.py \
  --year 2022 \
  --reference-root "$REFERENCE_ROOT" \
  --output-root "$DATASET_BASE/2022_AR" \
  --start-date 2022-01-01 \
  --end-date 2023-01-01 \
  --cloud-cover-max 20 \
  --inventory-only
```

Review the reported tile, STAC-item, and tile-date counts before downloading.

### 4.2 Download Sentinel-2 data

Remove `--inventory-only` to write the GeoTIFF files:

```bash
python scripts/download_ar_sentinel_stac.py \
  --year 2022 \
  --reference-root "$REFERENCE_ROOT" \
  --output-root "$DATASET_BASE/2022_AR" \
  --start-date 2022-01-01 \
  --end-date 2023-01-01 \
  --cloud-cover-max 20 \
  --workers 4
```

To fill or verify the archived 2023 directory using its existing grids:

```bash
python scripts/download_ar_sentinel_stac.py \
  --year 2023 \
  --reference-root "$REFERENCE_ROOT" \
  --output-root "$DATASET_BASE/2023_AR" \
  --start-date 2023-01-01 \
  --end-date 2024-01-01 \
  --cloud-cover-max 20 \
  --workers 4
```

Downloads are resumable. A valid existing raster with the expected grid is
skipped. Each yearly output also receives a `download_manifest.json`.

### 4.3 Download and align CDL crop masks

Create a crop mask using the CDL published for each training/evaluation year:

```bash
python scripts/prepare_ar_cdl_proxy.py \
  --cdl-year 2022 \
  --target-years 2022 \
  --dataset-base "$DATASET_BASE" \
  --reference-root "$REFERENCE_ROOT" \
  --workers 4

python scripts/prepare_ar_cdl_proxy.py \
  --cdl-year 2023 \
  --target-years 2023 \
  --dataset-base "$DATASET_BASE" \
  --reference-root "$REFERENCE_ROOT" \
  --workers 4
```

The script downloads one regional CDL raster and resamples it with
nearest-neighbor interpolation onto every tile's reference B4 grid.

To reuse one year's CDL as a proxy for another target year, pass the target
year to `--target-years`.

### 4.4 Validate every raster grid

Check that the existing 2022 and 2023 rasters are readable and aligned to the
2023 reference grid. The validation covers tile names, required bands, CDL
files, image dimensions, CRS, and affine transforms:

```bash
python scripts/validate_ar_current_data.py \
  --years 2022 2023 \
  --dataset-base "$DATASET_BASE" \
  --reference-root "$REFERENCE_ROOT" \
  --out-json "$DATASET_BASE/validation_2022_2023.json"
```

The command exits with an error if a tile, band, CDL file, or raster grid does
not match the reference. It does not check whether every expected acquisition
date was downloaded, or validate pixel values, clouds, or image content.

## 5. Create the Excel Inputs for Training and Inference

The training and inference code does not read the downloaded GeoTIFF files
directly. First, convert them into one Excel workbook for each tile:

```text
outputs/<YEAR>_AR/<TILE>/harvest_summary_all_crops.xlsx
```

To create the required workbooks for both 2022 and 2023, run:

```bash
WORKBOOK_ONLY=1 \
  bash scripts/run_prepare_doy_inputs_2022_2023.sh 8
```

Here, `8` is the number of tiles processed in parallel. This command creates
the Excel inputs required by both model training and inference.

To create workbooks for only one year, specify the year before the worker
count:

```bash
WORKBOOK_ONLY=1 \
  bash scripts/run_prepare_doy_inputs_parallel.sh 2022 8
```

The following full-analysis mode is optional and is not needed for model
training or inference. Use it only when you also need farm-level summaries and
the per-date NDVI/NDWI/EVI NPY and PNG files:

```bash
bash scripts/run_prepare_doy_inputs_parallel.sh 2022 8
```

Generated workbooks and intermediate outputs are written below `outputs/` and
are ignored by Git. Verify that workbooks were produced before training:

```bash
find outputs/2022_AR -name harvest_summary_all_crops.xlsx | head
find outputs/2023_AR -name harvest_summary_all_crops.xlsx | head
```

The processing scripts maintain per-tile checkpoints and can resume interrupted
runs. When using `run_prepare_doy_inputs_parallel.sh` directly, set
`RESET_CHECKPOINTS=0` to resume an existing run. The two-year convenience
wrapper intentionally resets checkpoints and overwrites its generated outputs.

## 6. Run the Tests

Run the unit tests before starting a long GPU job:

```bash
pytest -q doy_prediction/tests
```

These tests cover time-series tensor construction and the tile-radius utilities.
They do not download satellite data or train the full models.

## 7. Run a Small Training Check

Before launching the all-crops sweep, train a short CNN example for Rice:

```bash
python -m doy_prediction.train_tile_cnn \
  --outputs-root outputs \
  --crops Rice \
  --feature-set all_indices \
  --train-years 2022 \
  --test-years 2023 \
  --save-dir outputs_prediction_DOY/models/tutorial_cnn_1year \
  --epochs 2 \
  --batch-size 32 \
  --num-workers 2 \
  --device cuda
```

This command:

- trains on 2022 tile/crop records;
- takes the validation split from the 2022 records;
- selects the checkpoint with the best validation mean MAE;
- evaluates the selected checkpoint on 2023; and
- writes the checkpoint, metrics, and prediction CSV files under
  `outputs_prediction_DOY/models/tutorial_cnn_1year/all_indices/Rice/`.

The same interface is available for:

```text
python -m doy_prediction.train_tile_rnn
python -m doy_prediction.train_tile_hybrid
```

Use `python -m <module> --help` to see model-specific options.

## 8. Train and Evaluate the Full Model Sweep

The maintained sweep trains all crops listed in
`configs/Arkansas/gt_windows.yaml` for:

- CNN, RNN, and end-to-end hybrid models;
- `all_indices` (NDVI, NDWI, EVI) and `ndvi_only`;
- `6mo`, `9mo`, and `1year` input configurations; and
- 2022 training with 2023 evaluation.

It also runs late-fusion hybrid inference from the trained CNN and RNN
checkpoints. The complete default sweep contains hundreds of GPU training jobs.

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

The runner skips an existing model when both `metrics.json` and
`best_model.pt` are present. Set `RESUME_EXISTING=0` only when a deliberate
rerun is required.

Weights & Biases logging is optional. The examples disable it so a first run
does not require an account or API key. If W&B is installed and configured,
set `WANDB_ENABLED=1` and choose `WANDB_MODE=online` or `offline`.

### Temporal-window configuration

The `6mo` and `9mo` runs use:

```text
configs/Arkansas/doy_input_first_6mo.yaml
configs/Arkansas/doy_input_first_9mo.yaml
```

A crop listed in one of these files is restricted to that date range. A crop
that is not listed uses all available dates. The `1year` run always uses all
available dates.

## 9. Read and Aggregate Evaluation Results

Training writes evaluation results into each crop's `metrics.json`.

| Metric | Meaning | Better |
|---|---|---|
| `mae_start` | Mean absolute error of harvest start DOY, in days | Lower |
| `mae_end` | Mean absolute error of harvest end DOY, in days | Lower |
| `mae_mean` | Mean of start and end MAE | Lower |
| `mae_window` | MAE of predicted harvest-window length, in days | Lower |
| `iou_mean` | Mean intersection-over-union of predicted and target windows | Higher |
| `ordered_pct` | Fraction with predicted start DOY no later than end DOY | Higher |

Create one comparison workbook after the sweep completes:

```bash
export RESULTS_ROOT="$REPO_ROOT/outputs_prediction_DOY/models/$RUN_LABEL"

python -m doy_prediction.aggregate_all_crops_window_results \
  --results-root "$RESULTS_ROOT" \
  --out-xlsx "$RESULTS_ROOT/all_crops_doy_window_summary.xlsx" \
  --train-year 2022 \
  --test-year 2023
```

Check `run_status.tsv` for failed jobs before interpreting the workbook.

## 10. Run Inference with Existing CNN and RNN Checkpoints

The archived model can be used without retraining after Git LFS has downloaded
the checkpoints and model-input workbooks have been prepared.

```bash
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

## 11. Reproduce the Tested 2025 Workflow

The 2025 runner performs the complete workflow used for the archived inference
results:

1. Download 2025 Sentinel-2 data.
2. Prepare the 2025 CDL for every tile.
3. Validate every raster grid.
4. Build workbook-only inputs.
5. Run all three archived windows and both feature sets for 2025.

Use the activated Conda environment for both data processing and model
inference:

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

The archived inference directory is stored in Git and contains the tested 2025
predictions for the `6mo`, `9mo`, and `1year` windows with both `all_indices`
and `ndvi_only` features.

## Configuration Reference

| File | Purpose |
|---|---|
| `configs/Arkansas/cdl.yaml` | CDL numeric labels and crop names |
| `configs/Arkansas/gt_windows.yaml` | Target harvest windows and all-crop sweep list |
| `configs/Arkansas/seeding_config.yaml` | Seeding-estimation windows and offsets |
| `configs/Arkansas/doy_input_first_6mo.yaml` | Optional January-June input windows |
| `configs/Arkansas/doy_input_first_9mo.yaml` | Optional January-September input windows |

## Common Problems

### `No reference B4 raster` or `No reference tiles found`

`REFERENCE_ROOT` is missing the archived tile structure. Confirm that it points
to a directory such as `$DATASET_BASE/2023_AR` and that every tile contains at
least one acquisition with B2, B4, B8, B11, and SCL rasters.

### `Conda environment python not found`

Activate the project environment before running a shell script:

```bash
conda activate deepsatmodels_env
```

For a differently named environment, set `CONDA_ENV_PREFIX` to its absolute
path.

### `No tile-crop records found`

Confirm that both yearly output directories contain
`harvest_summary_all_crops.xlsx` files and that the requested crop appears in
those workbooks.

### CUDA is unavailable

Check the driver and PyTorch from inside the activated environment:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

The maintained training workflow expects this command to print `True`.

### W&B credentials are requested

Disable optional W&B logging for a local run:

```bash
export WANDB_ENABLED=0
```

## License and Data Terms

Before redistributing raw Sentinel-2 imagery, CDL files, trained checkpoints,
or derived outputs, verify the applicable upstream data terms and the intended
sharing policy for this research project.
