## Train
Train crop-specific 1D-CNN models that predict harvest start/end DOY from tile-crop time series.

```bash
MKL_THREADING_LAYER=GNU python -m doy_prediction.train_tile_cnn --outputs-root ./outputs --feature-set ndvi_only --train-years 2019 2020 2021 2022 --test-years 2023 --save-dir ./outputs_prediction_DOY/models/harvest_cnn_trial --epochs 30 --batch-size 32 --device cuda --wandb --wandb-project DeepSatModels-harvest
```

RNN training uses the same data builder and arguments. Like the CNN, the RNN receives the fixed 73-bin tensor for every sample; first-6-month and first-9-month runs keep the zero-filled rest of the year in the input.

```bash
MKL_THREADING_LAYER=GNU python -m doy_prediction.train_tile_rnn --outputs-root ./outputs --feature-set ndvi_only --train-years 2019 2020 2021 2022 --test-years 2023 --save-dir ./outputs_prediction_DOY/models/harvest_rnn_trial --epochs 30 --batch-size 32 --device cuda --rnn-type gru --hidden-size 64 --num-layers 2
```

To use three RNN layers with increasing hidden dimensions like the CNN channels:

```bash
MKL_THREADING_LAYER=GNU python -m doy_prediction.train_tile_rnn --outputs-root ./outputs --feature-set ndvi_only --train-years 2019 2020 2021 2022 --test-years 2023 --save-dir ./outputs_prediction_DOY/models/harvest_rnn_trial --epochs 30 --batch-size 32 --device cuda --rnn-type gru --hidden-sizes 32 64 128
```

Common training arguments:

- `--outputs-root`: Root directory containing yearly output folders such as `2019_AR`, `2020_AR`, etc.
- `--feature-set`: Input features to use. Choices are `ndvi_only` or `all_indices`.
- `--train-years`: Years used for model training.
- `--test-years`: Years used for held-out testing.
- `--save-dir`: Directory where checkpoints, metrics, and prediction CSVs are written.
- `--epochs`: Number of training epochs.
- `--batch-size`: Training batch size.
- `--device`: PyTorch device, for example `cuda` or `cpu`.
- `--crops`: Optional crop list. Defaults to `Corn Rice Soybeans`.
- `--min-points`: Minimum number of observations required inside the selected input window. Defaults to `2`.
- `--crop-windows-yaml`: Optional YAML file that limits which dates are fed into the model.
- `--log-test-every-epoch`: Log `test/...` metrics every epoch in W&B so they can be charted like `val/...`. This does not affect model selection.
- `--wandb`: Enable Weights & Biases logging.
- `--wandb-project`: Weights & Biases project name.

RNN-specific training arguments:

- `--rnn-type`: Recurrent layer type. Choices are `gru`, `lstm`, or `rnn`.
- `--hidden-size`: Hidden state size.
- `--num-layers`: Number of recurrent layers.
- `--hidden-sizes`: Optional per-layer hidden sizes, such as `32 64 128`. When set, this overrides the uniform `--hidden-size`/`--num-layers` architecture.
- `--dropout`: Dropout used between recurrent layers and in the prediction head.
- `--bidirectional`: Use a bidirectional recurrent layer.

Crop-specific fixed windows:
```bash
MKL_THREADING_LAYER=GNU python -m doy_prediction.train_tile_cnn --outputs-root ./outputs --feature-set ndvi_only --train-years 2019 2020 2021 2022 --test-years 2023 --save-dir ./outputs_prediction_DOY/models/harvest_cnn_trial --epochs 30 --batch-size 32 --device cuda --crop-windows-yaml ./configs/Arkansas/doy_crop_input_windows.yaml
```

## Prediction
Predict harvest start/end DOY from trained crop-specific checkpoints.

```bash
python -m doy_prediction.predict_tile_cnn --inputs ./outputs/2023_AR --checkpoints ./models/harvest_cnn_trial --feature-set ndvi_only --out-csv ./predictions_rice.csv --crops Rice
```

RNN checkpoints are predicted with `predict_tile_rnn`:

```bash
python -m doy_prediction.predict_tile_rnn --inputs ./outputs/2023_AR --checkpoints ./models/harvest_rnn_trial --feature-set ndvi_only --out-csv ./predictions_rice.csv --crops Rice
```

Crop-specific fixed windows:
```bash
python -m doy_prediction.predict_tile_cnn --inputs ./outputs/2023_AR --checkpoints ./models/harvest_cnn_trial --feature-set ndvi_only --out-csv ./predictions_rice.csv --crops Rice --crop-windows-yaml ./configs/Arkansas/doy_crop_input_windows.yaml
```

The crop-window YAML uses the same style as `gt_windows.yaml` and accepts either `[MM-DD, MM-DD]` or `{start: MM-DD, end: MM-DD}` for each crop.
Window filtering keeps GT `Start`/`End` labels from the workbook, but only feeds observations inside the configured input window into the model.

Without `--crop-windows-yaml`, the model uses all observations available across the full year. Use this option when you want to train or predict with only part of the year, such as the first 6 months or first 9 months, while still keeping the workbook `Start`/`End` labels as the targets.

Ready-to-use YAML files are stored here:

- First 6 months: `./configs/Arkansas/doy_input_first_6mo.yaml`
- First 9 months: `./configs/Arkansas/doy_input_first_9mo.yaml`

`./configs/Arkansas/doy_input_first_6mo.yaml`:

```yaml
Corn: [01-01, 06-30]
Rice: [01-01, 06-30]
Soybeans: [01-01, 06-30]
```

`./configs/Arkansas/doy_input_first_9mo.yaml`:

```yaml
Corn: [01-01, 09-30]
Rice: [01-01, 09-30]
Soybeans: [01-01, 09-30]
```

Then add the YAML path to the train and prediction commands:

```bash
--crop-windows-yaml ./configs/Arkansas/doy_input_first_6mo.yaml
```

Prediction CSVs now also include `window_iou` when the input workbooks contain ground-truth harvest start/end labels. Training metrics now report `iou_mean` alongside the existing MAE values.
