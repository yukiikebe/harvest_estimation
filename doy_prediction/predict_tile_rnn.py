from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from doy_prediction.tile_cnn_data import build_tile_crop_records, load_crop_windows_yaml
from doy_prediction.metrics import maybe_harvest_window_iou
from doy_prediction.tile_cnn_model import normalized_to_doy
from doy_prediction.tile_rnn_model import TileRNNRegressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict harvest start/end DOY from tile-crop workbooks.")
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--feature-set", choices=["ndvi_only", "all_indices"], default="all_indices")
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--crops", nargs="*", default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--min-points", type=int, default=2)
    parser.add_argument(
        "--crop-windows-yaml",
        type=Path,
        default=None,
        help="Optional YAML mapping crop names to fixed input windows [MM-DD, MM-DD].",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crop_windows = load_crop_windows_yaml(args.crop_windows_yaml)
    records = build_tile_crop_records(
        args.inputs,
        crops=args.crops,
        feature_set=args.feature_set,
        min_points=args.min_points,
        require_labels=False,
        crop_windows=crop_windows,
    )
    if not records:
        raise RuntimeError("No records found for prediction.")

    rows = []
    for crop in sorted({record.crop for record in records}):
        crop_records = [record for record in records if record.crop == crop]
        checkpoint_path = args.checkpoints / args.feature_set / crop.replace("/", "_") / "best_model.pt"
        if not checkpoint_path.exists():
            print(f"[skip] Missing checkpoint for crop={crop}: {checkpoint_path}")
            continue

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_config = dict(checkpoint.get("model_config", {}))
        model = TileRNNRegressor(in_channels=int(checkpoint["in_channels"]), **model_config)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        model.eval()

        with torch.no_grad():
            x = torch.stack([torch.from_numpy(record.x) for record in crop_records]).to(
                device=device,
                dtype=torch.float32,
            )
            pred = model(x)
            pred_doy = normalized_to_doy(pred)

        for record, pred_row in zip(crop_records, pred_doy):
            pred_start_doy = int(pred_row[0])
            pred_end_doy = int(pred_row[1])
            window_iou = maybe_harvest_window_iou(
                pred_start_doy=pred_start_doy,
                pred_end_doy=pred_end_doy,
                true_start_doy=record.start_doy,
                true_end_doy=record.end_doy,
            )
            rows.append(
                {
                    "year": record.year,
                    "tile": record.tile,
                    "crop": record.crop,
                    "true_start_doy": blank_if_none(record.start_doy),
                    "true_end_doy": blank_if_none(record.end_doy),
                    "pred_start_doy": pred_start_doy,
                    "pred_end_doy": pred_end_doy,
                    "window_iou": blank_if_none(window_iou),
                    "feature_set": args.feature_set,
                    "source_workbook": record.source_workbook,
                }
            )

    if not rows:
        raise RuntimeError("No predictions were generated.")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} predictions to {args.out_csv}")


def blank_if_none(value: object) -> object:
    if value is None:
        return ""
    return value


if __name__ == "__main__":
    main()
