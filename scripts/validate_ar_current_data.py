#!/usr/bin/env python3
"""Validate current-year Arkansas inputs against the archived 2023 grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import rasterio

BANDS = ("B2", "B4", "B8", "B11", "SCL")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, required=True)
    parser.add_argument("--dataset-base", type=Path, required=True)
    parser.add_argument(
        "--reference-root",
        type=Path,
        required=True,
    )
    parser.add_argument("--out-json", type=Path, default=None)
    return parser.parse_args()


def find_reference(tile_dir: Path, band: str) -> Path:
    paths = sorted(tile_dir.glob(f"*/{band}_*.tif"))
    if not paths:
        raise FileNotFoundError(f"Missing reference {band} below {tile_dir}")
    return paths[0]


def signature(path: Path) -> tuple[int, int, str, tuple[float, ...]]:
    with rasterio.open(path) as src:
        return (
            src.width,
            src.height,
            str(src.crs),
            tuple(round(value, 9) for value in src.transform),
        )


def main() -> None:
    args = parse_args()
    reference_tiles = sorted(
        path for path in args.reference_root.iterdir() if path.is_dir()
    )
    reference_names = {path.name for path in reference_tiles}
    reference_signatures = {
        (tile.name, band): signature(find_reference(tile, band))
        for tile in reference_tiles
        for band in BANDS
    }
    reference_b4 = {
        tile.name: reference_signatures[(tile.name, "B4")] for tile in reference_tiles
    }

    report: dict[str, Any] = {
        "reference_root": args.reference_root.name,
        "reference_tile_count": len(reference_tiles),
        "years": {},
    }
    any_errors = False
    for year in args.years:
        root = args.dataset_base / f"{year}_AR"
        target_tiles = sorted(path for path in root.iterdir() if path.is_dir())
        target_names = {path.name for path in target_tiles}
        errors: list[str] = []
        missing_tiles = sorted(reference_names - target_names)
        extra_tiles = sorted(target_names - reference_names)
        if missing_tiles:
            errors.append(f"missing tiles: {missing_tiles}")
        if extra_tiles:
            errors.append(f"extra tiles: {extra_tiles}")

        date_count = 0
        all_dates: set[str] = set()
        raster_count = 0
        for tile in target_tiles:
            if tile.name not in reference_names:
                continue
            cdl_path = tile / "cdl.tif"
            if not cdl_path.exists():
                errors.append(f"{tile.name}: missing cdl.tif")
            else:
                try:
                    if signature(cdl_path) != reference_b4[tile.name]:
                        errors.append(f"{tile.name}: cdl.tif grid mismatch")
                except Exception as exc:  # noqa: BLE001 - report unreadable inputs
                    errors.append(f"{tile.name}: unreadable cdl.tif: {exc!r}")

            date_dirs = sorted(path for path in tile.iterdir() if path.is_dir())
            if not date_dirs:
                errors.append(f"{tile.name}: no acquisition directories")
            for date_dir in date_dirs:
                date_count += 1
                all_dates.add(date_dir.name)
                for band in BANDS:
                    raster_path = date_dir / f"{band}_{date_dir.name}.tif"
                    if not raster_path.exists():
                        errors.append(f"{tile.name}/{date_dir.name}: missing {band}")
                        continue
                    raster_count += 1
                    try:
                        if signature(raster_path) != reference_signatures[(tile.name, band)]:
                            errors.append(
                                f"{tile.name}/{date_dir.name}: {band} grid mismatch"
                            )
                    except Exception as exc:  # noqa: BLE001 - report unreadable inputs
                        errors.append(
                            f"{tile.name}/{date_dir.name}: unreadable {band}: {exc!r}"
                        )

        year_report = {
            "root": root.name,
            "tile_count": len(target_tiles),
            "missing_tile_count": len(missing_tiles),
            "extra_tile_count": len(extra_tiles),
            "tile_date_count": date_count,
            "unique_date_count": len(all_dates),
            "first_date": min(all_dates) if all_dates else None,
            "last_date": max(all_dates) if all_dates else None,
            "raster_count": raster_count,
            "error_count": len(errors),
            "errors": errors[:1000],
        }
        report["years"][str(year)] = year_report
        any_errors = any_errors or bool(errors)
        print(
            f"year={year} tiles={len(target_tiles)} tile_dates={date_count} "
            f"unique_dates={len(all_dates)} first={year_report['first_date']} "
            f"last={year_report['last_date']} rasters={raster_count} errors={len(errors)}"
        )

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote validation report: {args.out_json}")
    if any_errors:
        raise SystemExit("Current-year data validation failed")


if __name__ == "__main__":
    main()
