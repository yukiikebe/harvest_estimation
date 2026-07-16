from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


RADIUS_FIELDS = [
    "radius_x_crs",
    "radius_y_crs",
    "radius_corner_crs",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append per-tile radius columns to a tile center CSV.",
    )
    parser.add_argument(
        "--in-csv",
        type=Path,
        required=True,
        help="Input CSV produced by export_tile_centers.py.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        required=True,
        help="Output CSV with radius columns appended.",
    )
    return parser.parse_args()


def _get_size(row: dict[str, str], size_key: str, bound_min_key: str, bound_max_key: str) -> float:
    raw_size = row.get(size_key)
    if raw_size not in (None, ""):
        return float(raw_size)

    return float(row[bound_max_key]) - float(row[bound_min_key])


def compute_radii(row: dict[str, str]) -> dict[str, float]:
    tile_size_x = _get_size(row, "tile_size_x_crs", "bounds_left", "bounds_right")
    tile_size_y = _get_size(row, "tile_size_y_crs", "bounds_bottom", "bounds_top")
    radius_x = tile_size_x / 2.0
    radius_y = tile_size_y / 2.0

    return {
        "radius_x_crs": radius_x,
        "radius_y_crs": radius_y,
        "radius_corner_crs": math.hypot(radius_x, radius_y),
    }


def augment_rows_with_radii(rows: list[dict[str, str]]) -> list[dict[str, str | float]]:
    augmented_rows: list[dict[str, str | float]] = []
    for row in rows:
        new_row: dict[str, str | float] = dict(row)
        new_row.update(compute_radii(row))
        augmented_rows.append(new_row)
    return augmented_rows


def write_rows(out_csv: Path, rows: list[dict[str, str | float]], fieldnames: list[str]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    with args.in_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV has no header: {args.in_csv}")

        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    augmented_rows = augment_rows_with_radii(rows)
    output_fieldnames = fieldnames + [field for field in RADIUS_FIELDS if field not in fieldnames]
    write_rows(args.out_csv, augmented_rows, output_fieldnames)
    print(f"Saved {len(augmented_rows)} rows with radii to {args.out_csv}")


if __name__ == "__main__":
    main()
