from __future__ import annotations

import argparse
import csv
from pathlib import Path

import rasterio
from rasterio.warp import transform


FIELDNAMES = [
    "tile",
    "center_latitude",
    "center_longitude",
    "center_x_crs",
    "center_y_crs",
    "width_pixels",
    "height_pixels",
    "pixel_size_x",
    "pixel_size_y",
    "tile_size_x_crs",
    "tile_size_y_crs",
    "bounds_left",
    "bounds_bottom",
    "bounds_right",
    "bounds_top",
    "transform_a",
    "transform_b",
    "transform_c",
    "transform_d",
    "transform_e",
    "transform_f",
    "source_crs",
    "source_raster",
]

DEFAULT_SENTINEL_BAND = "B4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export tile center latitude/longitude and raster geometry to a CSV file.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root directory containing tile subdirectories (e.g. 0_0, 0_1, ...).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        required=True,
        help="Path to the output CSV file.",
    )
    parser.add_argument(
        "--tiles",
        nargs="*",
        default=None,
        help="Optional list of tile folder names to export.",
    )
    parser.add_argument(
        "--raster-name",
        type=str,
        default=None,
        help="Optional raster filename to use instead of the default first B4_*.tif inside each tile.",
    )
    return parser.parse_args()


def list_tile_dirs(dataset_root: Path, tiles: list[str] | None) -> list[Path]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"--dataset-root not found: {dataset_root}")
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"--dataset-root is not a directory: {dataset_root}")

    tile_dirs = sorted(path for path in dataset_root.iterdir() if path.is_dir())
    if tiles is None:
        return tile_dirs

    tile_lookup = {path.name: path for path in tile_dirs}
    missing = sorted(tile_name for tile_name in tiles if tile_name not in tile_lookup)
    if missing:
        raise FileNotFoundError(f"Requested tiles not found under {dataset_root}: {missing}")

    return [tile_lookup[tile_name] for tile_name in tiles]


def resolve_tile_raster(tile_dir: Path, raster_name: str | None) -> Path:
    if raster_name:
        direct_path = tile_dir / raster_name
        if direct_path.exists():
            return direct_path

        matches = sorted(path for path in tile_dir.rglob(raster_name) if path.is_file())
        if matches:
            return matches[0]

        raise FileNotFoundError(f"Raster not found for tile {tile_dir.name}: {tile_dir / raster_name}")

    for ts_dir in sorted(path for path in tile_dir.iterdir() if path.is_dir()):
        matches = sorted(
            path
            for path in ts_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".tif" and path.name.startswith(f"{DEFAULT_SENTINEL_BAND}_")
        )
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"No {DEFAULT_SENTINEL_BAND}_*.tif found under tile {tile_dir}. "
        "Use --raster-name to choose a different raster explicitly."
    )


def get_raster_center_lon_lat(raster_path: Path) -> tuple[float, float, str]:
    metadata = get_raster_metadata(raster_path)
    return metadata["center_longitude"], metadata["center_latitude"], metadata["source_crs"]


def get_raster_metadata(raster_path: Path) -> dict[str, object]:
    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError(f"Raster has no CRS: {raster_path}")

        center_x, center_y = src.transform * (src.width / 2.0, src.height / 2.0)
        lon_values, lat_values = transform(src.crs, "EPSG:4326", [center_x], [center_y])
        bounds = src.bounds
        pixel_size_x = abs(src.transform.a)
        pixel_size_y = abs(src.transform.e)

        return {
            "center_latitude": lat_values[0],
            "center_longitude": lon_values[0],
            "center_x_crs": center_x,
            "center_y_crs": center_y,
            "width_pixels": src.width,
            "height_pixels": src.height,
            "pixel_size_x": pixel_size_x,
            "pixel_size_y": pixel_size_y,
            "tile_size_x_crs": src.width * pixel_size_x,
            "tile_size_y_crs": src.height * pixel_size_y,
            "bounds_left": bounds.left,
            "bounds_bottom": bounds.bottom,
            "bounds_right": bounds.right,
            "bounds_top": bounds.top,
            "transform_a": src.transform.a,
            "transform_b": src.transform.b,
            "transform_c": src.transform.c,
            "transform_d": src.transform.d,
            "transform_e": src.transform.e,
            "transform_f": src.transform.f,
            "source_crs": str(src.crs),
            "source_raster": str(raster_path.resolve()),
        }


def build_tile_center_rows(
    dataset_root: Path,
    *,
    tiles: list[str] | None = None,
    raster_name: str | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for tile_dir in list_tile_dirs(dataset_root, tiles):
        raster_path = resolve_tile_raster(tile_dir, raster_name)
        metadata = get_raster_metadata(raster_path)
        row = {"tile": tile_dir.name}
        row.update({field: metadata[field] for field in FIELDNAMES if field != "tile"})
        rows.append(row)

    return rows


def write_tile_center_csv(out_csv: Path, rows: list[dict[str, object]]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = build_tile_center_rows(
        args.dataset_root,
        tiles=args.tiles,
        raster_name=args.raster_name,
    )
    write_tile_center_csv(args.out_csv, rows)
    print(f"Saved {len(rows)} tile centers to {args.out_csv}")


if __name__ == "__main__":
    main()
