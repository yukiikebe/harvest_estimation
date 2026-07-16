from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Tuple

import torch

from doy_prediction.metrics import maybe_harvest_window_iou
from doy_prediction.tile_cnn_data import (
    TileCropRecord,
    build_tile_crop_records,
    load_crop_windows_yaml,
)
from doy_prediction.tile_cnn_model import TileCNNRegressor, normalized_to_doy
from doy_prediction.tile_rnn_model import TileRNNRegressor


RecordKey = Tuple[int, str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid inference using CNN start-day predictions and RNN end-day predictions."
    )
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--cnn-checkpoints", type=Path, required=True)
    parser.add_argument("--rnn-checkpoints", type=Path, required=True)
    parser.add_argument(
        "--feature-set",
        choices=["ndvi_only", "all_indices"],
        default="all_indices",
        help="Default feature set for both CNN and RNN unless overridden.",
    )
    parser.add_argument("--cnn-feature-set", choices=["ndvi_only", "all_indices"], default=None)
    parser.add_argument("--rnn-feature-set", choices=["ndvi_only", "all_indices"], default=None)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--crops", nargs="*", default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--min-points", type=int, default=2)
    parser.add_argument(
        "--crop-windows-yaml",
        type=Path,
        default=None,
        help="Default crop-window YAML for both CNN and RNN unless overridden.",
    )
    parser.add_argument(
        "--cnn-crop-windows-yaml",
        type=Path,
        default=None,
        help="Optional crop-window YAML used only for CNN inputs.",
    )
    parser.add_argument(
        "--rnn-crop-windows-yaml",
        type=Path,
        default=None,
        help="Optional crop-window YAML used only for RNN inputs.",
    )
    parser.add_argument(
        "--no-enforce-order",
        action="store_true",
        help="Do not force hybrid end DOY to be at least the hybrid start DOY.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_feature_set = args.cnn_feature_set or args.feature_set
    rnn_feature_set = args.rnn_feature_set or args.feature_set
    cnn_windows = load_crop_windows_yaml(args.cnn_crop_windows_yaml or args.crop_windows_yaml)
    rnn_windows = load_crop_windows_yaml(args.rnn_crop_windows_yaml or args.crop_windows_yaml)

    cnn_records = build_tile_crop_records(
        args.inputs,
        crops=args.crops,
        feature_set=cnn_feature_set,
        min_points=args.min_points,
        require_labels=False,
        crop_windows=cnn_windows,
    )
    rnn_records = build_tile_crop_records(
        args.inputs,
        crops=args.crops,
        feature_set=rnn_feature_set,
        min_points=args.min_points,
        require_labels=False,
        crop_windows=rnn_windows,
    )
    if not cnn_records:
        raise RuntimeError("No CNN input records found for prediction.")
    if not rnn_records:
        raise RuntimeError("No RNN input records found for prediction.")

    cnn_by_key = records_by_key(cnn_records)
    rnn_by_key = records_by_key(rnn_records)
    common_keys = set(cnn_by_key) & set(rnn_by_key)
    if not common_keys:
        raise RuntimeError("No matching tile-crop records found between CNN and RNN inputs.")

    rows = []
    for crop in sorted({key[2] for key in common_keys}):
        crop_keys = sorted(key for key in common_keys if key[2] == crop)
        crop_cnn_records = [cnn_by_key[key] for key in crop_keys]
        crop_rnn_records = [rnn_by_key[key] for key in crop_keys]

        cnn_checkpoint_path = checkpoint_path(args.cnn_checkpoints, cnn_feature_set, crop)
        rnn_checkpoint_path = checkpoint_path(args.rnn_checkpoints, rnn_feature_set, crop)
        if not cnn_checkpoint_path.exists():
            print(f"[skip] Missing CNN checkpoint for crop={crop}: {cnn_checkpoint_path}")
            continue
        if not rnn_checkpoint_path.exists():
            print(f"[skip] Missing RNN checkpoint for crop={crop}: {rnn_checkpoint_path}")
            continue

        cnn_pred_doy = predict_cnn(cnn_checkpoint_path, crop_cnn_records, device)
        rnn_pred_doy = predict_rnn(rnn_checkpoint_path, crop_rnn_records, device)

        for key, cnn_record, rnn_record, cnn_row, rnn_row in zip(
            crop_keys,
            crop_cnn_records,
            crop_rnn_records,
            cnn_pred_doy,
            rnn_pred_doy,
        ):
            pred_start_doy = int(cnn_row[0])
            pred_end_doy = int(rnn_row[1])
            if not args.no_enforce_order:
                pred_end_doy = max(pred_start_doy, pred_end_doy)

            true_start_doy = cnn_record.start_doy if cnn_record.start_doy is not None else rnn_record.start_doy
            true_end_doy = cnn_record.end_doy if cnn_record.end_doy is not None else rnn_record.end_doy
            window_iou = maybe_harvest_window_iou(
                pred_start_doy=pred_start_doy,
                pred_end_doy=pred_end_doy,
                true_start_doy=true_start_doy,
                true_end_doy=true_end_doy,
            )
            rows.append(
                {
                    "year": key[0],
                    "tile": key[1],
                    "crop": key[2],
                    "true_start_doy": blank_if_none(true_start_doy),
                    "true_end_doy": blank_if_none(true_end_doy),
                    "pred_start_doy": pred_start_doy,
                    "pred_end_doy": pred_end_doy,
                    "window_iou": blank_if_none(window_iou),
                    "cnn_start_doy": int(cnn_row[0]),
                    "cnn_end_doy": int(cnn_row[1]),
                    "rnn_start_doy": int(rnn_row[0]),
                    "rnn_end_doy": int(rnn_row[1]),
                    "cnn_feature_set": cnn_feature_set,
                    "rnn_feature_set": rnn_feature_set,
                    "cnn_source_workbook": cnn_record.source_workbook,
                    "rnn_source_workbook": rnn_record.source_workbook,
                    "cnn_num_observations": cnn_record.num_observations,
                    "rnn_num_observations": rnn_record.num_observations,
                }
            )

    if not rows:
        raise RuntimeError("No hybrid predictions were generated.")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} hybrid predictions to {args.out_csv}")


def records_by_key(records: list[TileCropRecord]) -> dict[RecordKey, TileCropRecord]:
    return {(record.year, record.tile, record.crop): record for record in records}


def checkpoint_path(checkpoints_root: Path, feature_set: str, crop: str) -> Path:
    return checkpoints_root / feature_set / crop.replace("/", "_") / "best_model.pt"


def predict_cnn(
    checkpoint_path: Path,
    records: list[TileCropRecord],
    device: torch.device,
) -> object:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = dict(checkpoint.get("model_config", {}))
    model = TileCNNRegressor(in_channels=int(checkpoint["in_channels"]), **model_config)
    model.load_state_dict(checkpoint["model_state"])
    return predict_model(model, records, device)


def predict_rnn(
    checkpoint_path: Path,
    records: list[TileCropRecord],
    device: torch.device,
) -> object:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = dict(checkpoint.get("model_config", {}))
    model = TileRNNRegressor(in_channels=int(checkpoint["in_channels"]), **model_config)
    model.load_state_dict(checkpoint["model_state"])
    return predict_model(model, records, device)


@torch.no_grad()
def predict_model(
    model: torch.nn.Module,
    records: list[TileCropRecord],
    device: torch.device,
) -> object:
    model.to(device)
    model.eval()
    x = torch.stack([torch.from_numpy(record.x) for record in records]).to(
        device=device,
        dtype=torch.float32,
    )
    return normalized_to_doy(model(x))


def blank_if_none(value: object) -> object:
    if value is None:
        return ""
    return value


if __name__ == "__main__":
    main()
