# Arkansas Harvest DOY Estimation

This repository predicts crop harvest start and end day-of-year (DOY) from Sentinel-2 vegetation-index time series. The maintained workflow is designed for Arkansas and supports downloading Sentinel-2 and USDA Cropland Data Layer (CDL) data, building crop-level time-series workbooks, training CNN/RNN models, evaluating them on a held-out year, and applying archived models to newer years.

The reference experiment trains on 2022 and evaluates on 2023. Its checkpoints and summary workbooks are stored under:

```text
outputs_prediction_DOY/models/all_crops_doy_window_sweep_2022_train_2023_test_20260628_144732
```

Model checkpoints (`*.pt` and `*.pth`) are managed with Git LFS.

## Workflow Phases

Follow the phase guides in order for a new experiment, or open the relevant guide when using data or checkpoints that have already been prepared.

| Phase | Guide | Main result |
|---|---|---|
| 1. Data download | [Download and prepare the data](docs/data-download.md) | Sentinel-2 rasters, CDL masks, and per-tile model-input workbooks |
| 2. Training | [Train and evaluate the models](docs/training.md) | CNN, RNN, and hybrid checkpoints, metrics, and prediction files |
| 3. Inference | [Run inference with existing checkpoints](docs/inference.md) | Harvest-window prediction CSV files for a selected year |

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
create_doy_prediction_input/   Converts GeoTIFFs and CDL masks into model inputs
docs/                          Data download, training, and inference guides
doy_prediction/                Models, training, inference, aggregation, and tests
scripts/                       Reproducible data, training, and inference commands
outputs_prediction_DOY/models/ Archived 2022-train/2023-test model sweep
deepsatmodels_env_latest.yml   Exported Conda environment
```

Large satellite data, generated model inputs, logs, and unarchived experiment outputs are intentionally not stored in Git.

## Prerequisites

Use a Linux machine with:

- Git and Git LFS
- Conda or Miniconda
- An NVIDIA GPU and a compatible NVIDIA driver
- Enough local storage for multi-year Sentinel-2 GeoTIFF data
- Internet access to Earth Search and USDA CropScape
- An existing archived Arkansas reference grid, normally `2023_AR`

The repository does not contain the raw 2022/2023 Sentinel-2 dataset. Ask the project maintainer for the archived `2023_AR` reference dataset before trying to create a new aligned yearly dataset.

> The STAC downloader is not a grid bootstrapper. It reads the tile names, raster dimensions, CRS, and affine transforms from an existing reference year and reproduces that grid exactly.

## Clone the Repository

Install Git LFS before retrieving the archived checkpoints:

```bash
git lfs install
git clone --branch yuki https://github.com/yukiikebe/harvest_estimation.git
cd harvest_estimation
git lfs pull
```

Run all commands in this documentation from the repository root.

## Create the Conda Environment

Create the environment using the exported environment file. Supplying `--name` overrides the machine-specific prefix stored in the exported YAML.

```bash
conda env create \
  --name deepsatmodels_env \
  --file deepsatmodels_env_latest.yml

conda activate deepsatmodels_env
```

The run scripts automatically use the active Conda environment. Training and inference are intended to run on an NVIDIA GPU.

## Configuration Reference

| File | Purpose |
|---|---|
| `configs/Arkansas/cdl.yaml` | CDL numeric labels and crop names |
| `configs/Arkansas/gt_windows.yaml` | Target harvest windows and all-crop sweep list |
| `configs/Arkansas/seeding_config.yaml` | Seeding-estimation windows and offsets |
| `configs/Arkansas/doy_input_first_6mo.yaml` | Optional January-June input windows |
| `configs/Arkansas/doy_input_first_9mo.yaml` | Optional January-September input windows |
