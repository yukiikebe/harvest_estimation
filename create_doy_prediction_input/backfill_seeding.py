from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from create_doy_prediction_input.seeding_estimator import SeedingEstimator
from create_doy_prediction_input.summary import summarize_farm_harvest_dates, summarize_global_crop_outputs
from create_doy_prediction_input.utils.plot_and_csv import _write_crop_summary_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill seeding dates into existing output CSV files.")
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root output directory containing tile subdirectories, such as outputs/2019_AR.",
    )
    parser.add_argument(
        "--seeding-config-yaml",
        type=Path,
        default=None,
        help="Optional YAML mapping crop_name -> {window_start, window_end, offset_days}.",
    )
    parser.add_argument(
        "--tiles",
        nargs="*",
        default=None,
        help="Optional list of tile folder names to process.",
    )
    return parser.parse_args()


def parse_date(text: str) -> Optional[datetime]:
    text = (text or "").strip()
    if not text:
        return None
    return datetime.strptime(text, "%Y-%m-%d")


def parse_float(text: str) -> float:
    text = (text or "").strip()
    return float(text) if text else np.nan


def load_existing_summary(csv_path: Path, fallback_farm_id: Optional[str] = None) -> dict[str, object]:
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    include_farm_id_column = "FarmID" in fieldnames
    div_start_header = "div_start" if "div_start" in fieldnames else "div_result_start" if "div_result_start" in fieldnames else "div_start"
    div_end_header = "div_end" if "div_end" in fieldnames else "div_result_end" if "div_result_end" in fieldnames else "div_end"

    timestamps: list[datetime] = []
    ndvi: list[float] = []
    ndwi: list[float] = []
    evi: list[float] = []

    pred_start = None
    pred_end = None
    chosen_rule = ""
    div_start = ""
    div_end = ""
    iou = ""
    farm_id = fallback_farm_id

    for row in rows:
        date_value = parse_date(row.get("Date", ""))
        if date_value is None:
            continue

        if include_farm_id_column and row.get("FarmID"):
            farm_id = row["FarmID"]

        ndvi_value = parse_float(row.get("NDVI", ""))
        ndwi_value = parse_float(row.get("NDWI", ""))
        evi_value = parse_float(row.get("EVI", ""))
        if not (np.isnan(ndvi_value) and np.isnan(ndwi_value) and np.isnan(evi_value)):
            timestamps.append(date_value)
            ndvi.append(ndvi_value)
            ndwi.append(ndwi_value)
            evi.append(evi_value)

        harvest_flag = (row.get("Harvest") or "").strip()
        if harvest_flag == "Start":
            pred_start = date_value
            chosen_rule = (row.get("start_rule") or "").strip()
            div_start = (row.get(div_start_header) or "").strip()
            iou = (row.get("IoU") or "").strip()
        elif harvest_flag == "End":
            pred_end = date_value
            div_end = (row.get(div_end_header) or "").strip()

    return {
        "timestamps": timestamps,
        "ndvi": np.asarray(ndvi, dtype=float),
        "ndwi": np.asarray(ndwi, dtype=float),
        "evi": np.asarray(evi, dtype=float),
        "pred_start": pred_start,
        "pred_end": pred_end,
        "chosen_rule": chosen_rule,
        "div_start": div_start,
        "div_end": div_end,
        "iou": iou,
        "farm_id": farm_id,
        "include_farm_id_column": include_farm_id_column,
        "div_start_header": div_start_header,
        "div_end_header": div_end_header,
    }


def rewrite_summary_csv(csv_path: Path, crop_name: str, estimator: SeedingEstimator, fallback_farm_id: Optional[str] = None) -> bool:
    payload = load_existing_summary(csv_path, fallback_farm_id=fallback_farm_id)
    timestamps = payload["timestamps"]
    if not timestamps:
        return False

    seeding_estimate = estimator.estimate(
        timestamps,
        payload["ndvi"],
        payload["evi"],
        crop_name=crop_name,
    )

    _write_crop_summary_csv(
        csv_path=csv_path,
        crop_name=crop_name,
        timestamps=timestamps,
        ndvi=payload["ndvi"],
        ndwi=payload["ndwi"],
        evi=payload["evi"],
        pred_start=payload["pred_start"],
        pred_end=payload["pred_end"],
        chosen_rule=payload["chosen_rule"],
        div_start=payload["div_start"],
        div_end=payload["div_end"],
        iou=payload["iou"],
        summary_rows=None,
        farm_id=payload["farm_id"],
        seeding_estimate=seeding_estimate,
        div_start_header=payload["div_start_header"],
        div_end_header=payload["div_end_header"],
        include_farm_id_column=payload["include_farm_id_column"],
    )
    return True


def iter_tile_dirs(output_root: Path, tiles: Optional[list[str]]) -> list[Path]:
    all_tiles = sorted(path for path in output_root.iterdir() if path.is_dir())
    if not tiles:
        return all_tiles
    wanted = set(tiles)
    return [path for path in all_tiles if path.name in wanted]


def main() -> None:
    args = parse_args()
    estimator = SeedingEstimator.from_yaml(args.seeding_config_yaml)

    rewritten = 0
    for tile_dir in iter_tile_dirs(args.output_root, args.tiles):
        print(f"Processing tile: {tile_dir.name}")

        for crop_dir in sorted(path for path in tile_dir.iterdir() if path.is_dir()):
            crop_name = crop_dir.name
            crop_summary_path = crop_dir / f"{crop_name}_summary.csv"
            if crop_summary_path.exists() and rewrite_summary_csv(crop_summary_path, crop_name, estimator):
                rewritten += 1

            farms_root = crop_dir / "Farms"
            if not farms_root.is_dir():
                continue

            for farm_dir in sorted(path for path in farms_root.iterdir() if path.is_dir()):
                farm_summary_path = farm_dir / f"{crop_name}_{farm_dir.name}_summary.csv"
                if farm_summary_path.exists() and rewrite_summary_csv(
                    farm_summary_path,
                    crop_name,
                    estimator,
                    fallback_farm_id=farm_dir.name,
                ):
                    rewritten += 1

        summarize_farm_harvest_dates(tile_dir)
        summarize_global_crop_outputs(tile_dir)

    print(f"Finished backfill. Rewrote {rewritten} summary CSV files.")


if __name__ == "__main__":
    main()
