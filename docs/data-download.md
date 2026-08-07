# Phase 1: Data Download and Preparation

[Back to the project overview](../README.md) | [Next: Training](training.md)

This phase downloads Sentinel-2 L2A imagery and USDA Cropland Data Layer (CDL) masks, aligns them to the archived Arkansas reference grid, and creates the per-tile Excel workbooks consumed by training and inference.

Complete the shared [prerequisites and environment setup](../README.md#prerequisites) before starting this phase.

## Files Used in This Phase

The Python files run individual operations. The shell wrappers automate multi-year or parallel workflows.

| Task | File | Main output |
|---|---|---|
| Download Sentinel-2 | `scripts/download_ar_sentinel_stac.py` | B2, B4, B8, B11, and SCL GeoTIFF files |
| Download and align CDL | `scripts/prepare_ar_cdl_proxy.py` | One `cdl.tif` per tile |
| Validate downloaded data | `scripts/validate_ar_current_data.py` | Terminal validation result and optional JSON report |
| Create model-input workbooks | `create_doy_prediction_input/main.py` | `harvest_summary_all_crops.xlsx` per tile |
| Prepare one year in parallel | `scripts/run_prepare_doy_inputs_parallel.sh` | Per-tile workbooks and optional analysis outputs |
| Prepare 2022 and 2023 | `scripts/run_prepare_doy_inputs_2022_2023.sh` | Training and evaluation workbooks for both years |

Use a Python program when running or debugging one operation. Use a shell wrapper when preparing complete yearly inputs.

## Configure Data Paths

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

Each year contains spatial tile directories. Each acquisition-date directory contains four reflectance bands and the Sentinel-2 Scene Classification Layer (SCL). A tile-level `cdl.tif` supplies the crop labels.

If complete 2022 and 2023 directories already exist, skip to [Create the model-input workbooks](#create-the-model-input-workbooks).

## Download Sentinel-2 Data

The downloader queries public Sentinel-2 L2A COGs and maps them onto the exact grid of `REFERENCE_ROOT`. `--end-date` is exclusive.

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

Downloads are resumable. A valid existing raster with the expected grid is skipped. Each yearly output also receives a `download_manifest.json`.

## Download and Align CDL Crop Masks

Create a crop mask using the CDL published for each training and evaluation year:

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

The script downloads one regional CDL raster and resamples it with nearest-neighbor interpolation onto every tile's reference B4 grid. To reuse one year's CDL as a proxy for another target year, pass the target year to `--target-years`.

## Create the Model-Input Workbooks

The training and inference code does not read the downloaded GeoTIFF files directly. First, convert them into one Excel workbook for each tile:

```text
outputs/<YEAR>_AR/<TILE>/harvest_summary_all_crops.xlsx
```

The workbook generator is `create_doy_prediction_input/main.py`. For example, the following command processes one year sequentially:

```bash
python -m create_doy_prediction_input.main \
  --dataset-root "$DATASET_BASE/2022_AR" \
  --cdl-yaml configs/Arkansas/cdl.yaml \
  --gt-windows-yaml configs/Arkansas/gt_windows.yaml \
  --seeding-config-yaml configs/Arkansas/seeding_config.yaml \
  --output-root outputs/2022_AR \
  --all-crops \
  --no-farm \
  --no-index-images
```

To create the required workbooks for both 2022 and 2023, run:

```bash
WORKBOOK_ONLY=1 \
  bash scripts/run_prepare_doy_inputs_2022_2023.sh 8
```

Here, `8` is the number of tiles processed in parallel. To create workbooks for only one year, specify the year before the worker count:

```bash
WORKBOOK_ONLY=1 \
  bash scripts/run_prepare_doy_inputs_parallel.sh 2022 8
```

The following full-analysis mode is optional and is not needed for model training or inference. Use it only when you also need farm-level summaries and the per-date NDVI/NDWI/EVI NPY and PNG files:

```bash
bash scripts/run_prepare_doy_inputs_parallel.sh 2022 8
```

Generated workbooks and intermediate outputs are written below `outputs/` and are ignored by Git.

The processing scripts maintain per-tile checkpoints and can resume interrupted runs. When using `run_prepare_doy_inputs_parallel.sh` directly, set `RESET_CHECKPOINTS=0` to resume an existing run. The two-year convenience wrapper intentionally resets checkpoints and overwrites its generated outputs.

Continue with [Phase 2: Training](training.md) after the required yearly workbooks are ready.
