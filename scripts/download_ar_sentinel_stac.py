#!/usr/bin/env python3
"""Download Arkansas Sentinel-2 L2A data on the archived 2022/2023 tile grid.

The downloader uses the public Element84 Earth Search STAC API and public
Sentinel-2 COG assets. Every output raster is reprojected onto the exact grid
of the corresponding band in a reference year (2023 by default). Downloads
are resumable: valid existing rasters are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds

STAC_SEARCH_URL = "https://earth-search.aws.element84.com/v1/search"
BAND_ASSETS = {
    "B2": "blue",
    "B4": "red",
    "B8": "nir",
    "B11": "swir16",
    "SCL": "scl",
}
REFLECTANCE_BANDS = frozenset({"B2", "B4", "B8", "B11"})


@dataclass(frozen=True)
class GridSpec:
    crs: str
    transform: rasterio.Affine
    width: int
    height: int
    bounds: tuple[float, float, float, float]


@dataclass(frozen=True)
class TileSpec:
    name: str
    grids: dict[str, GridSpec]

    @property
    def footprint(self) -> tuple[float, float, float, float]:
        return self.grids["B4"].bounds


@dataclass(frozen=True)
class DownloadTask:
    tile: TileSpec
    acquisition_date: str
    items: tuple[dict[str, Any], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Sentinel-2 L2A data on an existing Arkansas tile grid."
    )
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument(
        "--reference-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    parser.add_argument("--start-date", default=None, help="Inclusive YYYY-MM-DD.")
    parser.add_argument("--end-date", default=None, help="Exclusive YYYY-MM-DD.")
    parser.add_argument("--cloud-cover-max", type=float, default=20.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tiles", nargs="*", default=None)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


def find_reference_raster(tile_dir: Path, band: str) -> Path:
    candidates = sorted(tile_dir.glob(f"*/{band}_*.tif"))
    if not candidates:
        raise FileNotFoundError(f"No reference {band} raster below {tile_dir}")
    return candidates[0]


def read_grid(path: Path) -> GridSpec:
    with rasterio.open(path) as src:
        if src.count != 1:
            raise ValueError(f"Expected a single-band reference raster: {path}")
        return GridSpec(
            crs=str(src.crs),
            transform=src.transform,
            width=src.width,
            height=src.height,
            bounds=tuple(float(value) for value in src.bounds),
        )


def load_tiles(reference_root: Path, selected: Iterable[str] | None) -> list[TileSpec]:
    selected_set = set(selected) if selected else None
    tile_dirs = sorted(path for path in reference_root.iterdir() if path.is_dir())
    if selected_set is not None:
        found = {path.name for path in tile_dirs}
        missing = selected_set - found
        if missing:
            raise FileNotFoundError(f"Reference tiles not found: {sorted(missing)}")
        tile_dirs = [path for path in tile_dirs if path.name in selected_set]

    tiles: list[TileSpec] = []
    for tile_dir in tile_dirs:
        grids = {band: read_grid(find_reference_raster(tile_dir, band)) for band in BAND_ASSETS}
        if any(grid.crs != grids["B4"].crs for grid in grids.values()):
            raise ValueError(f"Mixed CRS in reference tile {tile_dir.name}")
        tiles.append(TileSpec(name=tile_dir.name, grids=grids))
    if not tiles:
        raise RuntimeError(f"No reference tiles found below {reference_root}")
    return tiles


def union_bbox_wgs84(tiles: list[TileSpec]) -> tuple[float, float, float, float]:
    bounds = [tile.footprint for tile in tiles]
    union = (
        min(value[0] for value in bounds),
        min(value[1] for value in bounds),
        max(value[2] for value in bounds),
        max(value[3] for value in bounds),
    )
    return tuple(
        float(value)
        for value in transform_bounds(tiles[0].grids["B4"].crs, "EPSG:4326", *union)
    )


def request_with_retries(
    session: requests.Session,
    method: str,
    url: str,
    *,
    retries: int,
    **kwargs: Any,
) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.request(method, url, timeout=120, **kwargs)
            response.raise_for_status()
            return response
        except (requests.RequestException, OSError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2 ** attempt, 15))
    raise RuntimeError(f"Request failed after {retries} attempts: {url}") from last_error


def search_items(
    bbox: tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    cloud_cover_max: float,
    retries: int,
) -> list[dict[str, Any]]:
    session = requests.Session()
    session.headers.update({"User-Agent": "DeepSatModels-current-data/1.0"})
    body: dict[str, Any] = {
        "collections": ["sentinel-2-l2a"],
        "bbox": list(bbox),
        "datetime": f"{start_date}T00:00:00Z/{end_date}T00:00:00Z",
        "limit": 100,
        "query": {"eo:cloud_cover": {"lt": cloud_cover_max}},
    }
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    while True:
        response = request_with_retries(
            session,
            "POST",
            STAC_SEARCH_URL,
            retries=retries,
            json=body,
        )
        payload = response.json()
        for item in payload.get("features", []):
            item_id = str(item["id"])
            if item_id not in seen:
                seen.add(item_id)
                items.append(item)
        next_link = next(
            (link for link in payload.get("links", []) if link.get("rel") == "next"),
            None,
        )
        if next_link is None:
            break
        body = dict(next_link.get("body") or body)
    items.sort(key=lambda item: (item["properties"]["datetime"], item["id"]))
    return items


def boxes_intersect(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> bool:
    return not (
        left[2] <= right[0]
        or left[0] >= right[2]
        or left[3] <= right[1]
        or left[1] >= right[3]
    )


def build_tasks(tiles: list[TileSpec], items: list[dict[str, Any]]) -> list[DownloadTask]:
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    target_crs = tiles[0].grids["B4"].crs
    for item in items:
        scene_bounds = tuple(
            float(value)
            for value in transform_bounds("EPSG:4326", target_crs, *item["bbox"])
        )
        acquisition_date = str(item["properties"]["datetime"])[:10]
        for tile in tiles:
            if boxes_intersect(tile.footprint, scene_bounds):
                by_key[(tile.name, acquisition_date)].append(item)

    tile_by_name = {tile.name: tile for tile in tiles}
    tasks = []
    for (tile_name, acquisition_date), task_items in sorted(by_key.items()):
        task_items.sort(
            key=lambda item: (
                float(item["properties"].get("eo:cloud_cover", 100.0)),
                str(item["id"]),
            )
        )
        tasks.append(
            DownloadTask(
                tile=tile_by_name[tile_name],
                acquisition_date=acquisition_date,
                items=tuple(task_items),
            )
        )
    return tasks


def raster_matches(path: Path, grid: GridSpec) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with rasterio.open(path) as src:
            return (
                src.count == 1
                and src.width == grid.width
                and src.height == grid.height
                and str(src.crs) == grid.crs
                and src.transform.almost_equals(grid.transform)
            )
    except (OSError, rasterio.errors.RasterioError):
        return False


def harmonize_reflectance(values: np.ndarray, item: dict[str, Any]) -> np.ndarray:
    properties = item.get("properties", {})
    # Earth Search COGs with this flag already have the -0.1 BOA offset
    # applied to their stored pixels and match the GEE HARMONIZED DN range.
    if properties.get("earthsearch:boa_offset_applied", False):
        return values
    try:
        processing_baseline = float(properties.get("s2:processing_baseline", "0"))
    except (TypeError, ValueError):
        processing_baseline = 0.0
    if processing_baseline < 4.0:
        return values
    adjusted = values.astype(np.int32, copy=False) - 1000
    adjusted[values == 0] = 0
    np.maximum(adjusted, 0, out=adjusted)
    return adjusted.astype(np.uint16)


def read_item_to_grid(
    href: str,
    grid: GridSpec,
    *,
    dtype: np.dtype[Any],
    resampling: Resampling,
) -> np.ndarray:
    destination = np.zeros((grid.height, grid.width), dtype=dtype)
    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
        GDAL_HTTP_MULTIRANGE="YES",
        GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
        VSI_CACHE="TRUE",
        VSI_CACHE_SIZE=50_000_000,
        GDAL_HTTP_MAX_RETRY="5",
        GDAL_HTTP_RETRY_DELAY="2",
    ), rasterio.open(href) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata if src.nodata is not None else 0,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            dst_nodata=0,
            resampling=resampling,
            init_dest_nodata=True,
            num_threads=1,
        )
    return destination


def write_raster(path: Path, values: np.ndarray, grid: GridSpec) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".part")
    profile: dict[str, Any] = {
        "driver": "GTiff",
        "width": grid.width,
        "height": grid.height,
        "count": 1,
        "dtype": str(values.dtype),
        "crs": grid.crs,
        "transform": grid.transform,
        "nodata": 0,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "BIGTIFF": "IF_SAFER",
    }
    if values.dtype != np.uint8:
        profile["predictor"] = 2
    with rasterio.open(temp_path, "w", **profile) as dst:
        dst.write(values, 1)
    os.replace(temp_path, path)


def download_task(task: DownloadTask, output_root: Path) -> tuple[str, str, int]:
    date_dir = output_root / task.tile.name / task.acquisition_date
    written = 0
    for band, asset_name in BAND_ASSETS.items():
        grid = task.tile.grids[band]
        output_path = date_dir / f"{band}_{task.acquisition_date}.tif"
        if raster_matches(output_path, grid):
            continue

        dtype = np.uint8 if band == "SCL" else np.uint16
        resampling = Resampling.nearest if band == "SCL" else Resampling.bilinear
        merged = np.zeros((grid.height, grid.width), dtype=dtype)
        found_asset = False
        for item in task.items:
            asset = item.get("assets", {}).get(asset_name)
            if not asset or not str(asset.get("href", "")).startswith("http"):
                continue
            values = read_item_to_grid(
                str(asset["href"]),
                grid,
                dtype=dtype,
                resampling=resampling,
            )
            if band in REFLECTANCE_BANDS:
                values = harmonize_reflectance(values, item)
            fill = (merged == 0) & (values != 0)
            merged[fill] = values[fill]
            found_asset = True

        if not found_asset:
            raise RuntimeError(
                f"No {asset_name} asset for tile={task.tile.name} date={task.acquisition_date}"
            )
        write_raster(output_path, merged, grid)
        written += 1

    metadata_path = date_dir / "stac_scenes.json"
    metadata = {
        "date": task.acquisition_date,
        "tile": task.tile.name,
        "scene_ids": [item["id"] for item in task.items],
        "scene_cloud_cover": [
            item["properties"].get("eo:cloud_cover") for item in task.items
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return task.tile.name, task.acquisition_date, written


def download_task_with_retries(
    task: DownloadTask,
    output_root: Path,
    retries: int,
) -> tuple[str, str, int]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return download_task(task, output_root)
        except Exception as exc:  # noqa: BLE001 - retry any per-task failure
            last_error = exc
            if attempt < retries:
                time.sleep(min(2 ** attempt, 15))
    raise RuntimeError(
        f"tile={task.tile.name} date={task.acquisition_date} failed after {retries} attempts"
    ) from last_error


def date_defaults(year: int) -> tuple[str, str]:
    start = f"{year:04d}-01-01"
    today = datetime.now(timezone.utc).date()
    if year == today.year:
        end = (today + timedelta(days=1)).isoformat()
    else:
        end = f"{year + 1:04d}-01-01"
    return start, end


def main() -> None:
    args = parse_args()
    default_start, default_end = date_defaults(args.year)
    start_date = args.start_date or default_start
    end_date = args.end_date or default_end
    output_root = args.output_root

    tiles = load_tiles(args.reference_root, args.tiles)
    bbox = union_bbox_wgs84(tiles)
    items = search_items(
        bbox,
        start_date,
        end_date,
        args.cloud_cover_max,
        args.retries,
    )
    tasks = build_tasks(tiles, items)
    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]

    print(f"year={args.year}")
    print(f"date_range=[{start_date}, {end_date})")
    print(f"reference_root={args.reference_root}")
    print(f"output_root={output_root}")
    print(f"tiles={len(tiles)}")
    print(f"region_bbox_wgs84={bbox}")
    print(f"stac_items={len(items)}")
    print(f"tile_date_tasks={len(tasks)}")
    if args.inventory_only:
        return

    output_root.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, str]] = []
    written_files = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_task_with_retries,
                task,
                output_root,
                args.retries,
            ): task
            for task in tasks
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            try:
                _, _, written = future.result()
                written_files += written
            except Exception as exc:  # noqa: BLE001 - collect per-task failures
                failures.append(
                    {
                        "tile": task.tile.name,
                        "date": task.acquisition_date,
                        "error": repr(exc),
                    }
                )
                print(
                    f"[failed] tile={task.tile.name} date={task.acquisition_date}: {exc}",
                    flush=True,
                )
            if completed % 25 == 0 or completed == len(tasks):
                print(
                    f"progress={completed}/{len(tasks)} written_files={written_files} "
                    f"failures={len(failures)}",
                    flush=True,
                )

    manifest = {
        "year": args.year,
        "start_date_inclusive": start_date,
        "end_date_exclusive": end_date,
        "cloud_cover_max": args.cloud_cover_max,
        "reference_root": str(args.reference_root.resolve()),
        "output_root": str(output_root.resolve()),
        "region_bbox_wgs84": bbox,
        "tile_count": len(tiles),
        "stac_item_count": len(items),
        "tile_date_task_count": len(tasks),
        "written_files_this_run": written_files,
        "failure_count": len(failures),
        "failures": failures,
    }
    (output_root / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    if failures:
        raise SystemExit(f"Download completed with {len(failures)} failed tile-date tasks")


if __name__ == "__main__":
    main()
