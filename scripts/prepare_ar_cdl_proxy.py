#!/usr/bin/env python3
"""Download Arkansas CDL and place aligned proxy masks in yearly tile roots."""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds

CDL_SERVICE_URL = (
    "https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Arkansas CDL and align it to existing Sentinel tile grids."
    )
    parser.add_argument("--cdl-year", type=int, default=2025)
    parser.add_argument("--target-years", nargs="+", type=int, required=True)
    parser.add_argument("--dataset-base", type=Path, required=True)
    parser.add_argument(
        "--reference-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--cdl-url",
        default=None,
        help="Optional direct GeoTIFF URL. By default CropScape is queried for the exact reference-grid union bbox.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tiles", nargs="*", default=None)
    parser.add_argument("--download-only", action="store_true")
    return parser.parse_args()


def download_file(url: str, output_path: Path, retries: int = 4) -> None:
    if output_path.is_file() and output_path.stat().st_size > 0:
        print(f"CDL already exists: {output_path}")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=180) as response:
                response.raise_for_status()
                with open(temp_path, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            os.replace(temp_path, output_path)
            print(f"Downloaded CDL: {output_path} ({output_path.stat().st_size} bytes)")
            return
        except (requests.RequestException, OSError) as exc:
            last_error = exc
            if temp_path.exists():
                temp_path.unlink()
            if attempt < retries:
                time.sleep(min(2 ** attempt, 15))
    raise RuntimeError(f"Failed to download {url}") from last_error


def reference_b4(tile_dir: Path) -> Path:
    paths = sorted(tile_dir.glob("*/B4_*.tif"))
    if not paths:
        raise FileNotFoundError(f"No B4 reference raster below {tile_dir}")
    return paths[0]


def mask_matches(mask_path: Path, reference_path: Path, cdl_path: Path) -> bool:
    if not mask_path.is_file() or mask_path.stat().st_size == 0:
        return False
    try:
        with rasterio.open(mask_path) as mask, rasterio.open(reference_path) as ref:
            return (
                mask.count == 1
                and mask.width == ref.width
                and mask.height == ref.height
                and str(mask.crs) == str(ref.crs)
                and mask.transform.almost_equals(ref.transform)
                and mask.tags().get("source_cdl") == str(cdl_path)
            )
    except (OSError, rasterio.errors.RasterioError):
        return False


def align_one_mask(cdl_path: Path, reference_path: Path, output_path: Path) -> int:
    if mask_matches(output_path, reference_path, cdl_path):
        return 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".part")
    with rasterio.open(reference_path) as ref, rasterio.open(cdl_path) as src:
        output = np.zeros((ref.height, ref.width), dtype=np.uint8)
        reproject(
            source=rasterio.band(src, 1),
            destination=output,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=ref.transform,
            dst_crs=ref.crs,
            dst_nodata=0,
            resampling=Resampling.nearest,
            init_dest_nodata=True,
            num_threads=1,
        )
        profile: dict[str, Any] = {
            "driver": "GTiff",
            "width": ref.width,
            "height": ref.height,
            "count": 1,
            "dtype": "uint8",
            "crs": ref.crs,
            "transform": ref.transform,
            "nodata": 0,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }
        with rasterio.open(temp_path, "w", **profile) as dst:
            dst.write(output, 1)
            dst.update_tags(source_cdl=str(cdl_path), proxy_cdl_year=cdl_path.stem)
    os.replace(temp_path, output_path)
    return 1


def resolve_region_cdl_url(
    cdl_year: int,
    reference_root: Path,
    direct_url: str | None,
) -> tuple[str, tuple[float, float, float, float]]:
    tile_dirs = sorted(path for path in reference_root.iterdir() if path.is_dir())
    bounds = []
    for tile_dir in tile_dirs:
        with rasterio.open(reference_b4(tile_dir)) as src:
            bounds.append(tuple(src.bounds))
            source_crs = str(src.crs)
    union = (
        min(value[0] for value in bounds),
        min(value[1] for value in bounds),
        max(value[2] for value in bounds),
        max(value[3] for value in bounds),
    )
    cdl_bbox = tuple(
        float(value)
        for value in transform_bounds(source_crs, "EPSG:5070", *union, densify_pts=21)
    )
    if direct_url:
        return direct_url, cdl_bbox

    response = None
    last_error: Exception | None = None
    for attempt in range(1, 5):
        try:
            response = requests.get(
                CDL_SERVICE_URL,
                params={
                    "year": cdl_year,
                    "bbox": ",".join(f"{value:.3f}" for value in cdl_bbox),
                },
                timeout=180,
            )
            response.raise_for_status()
            break
        except requests.RequestException as exc:
            last_error = exc
            if attempt < 4:
                time.sleep(min(2 ** attempt, 15))
    if response is None or not response.ok:
        raise RuntimeError("CropScape CDL request failed after 4 attempts") from last_error
    root = ElementTree.fromstring(response.text)
    return_url = next(
        (node.text for node in root.iter() if node.tag.endswith("returnURL")),
        None,
    )
    if not return_url:
        raise RuntimeError(f"CropScape did not return a CDL URL: {response.text[:500]}")
    return return_url, cdl_bbox


def main() -> None:
    args = parse_args()
    cdl_path = args.dataset_base / f"CDL_{args.cdl_year}_reference_region.tif"
    if cdl_path.is_file() and cdl_path.stat().st_size > 0 and args.cdl_url is None:
        cdl_url = "existing local reference-region CDL"
        _, cdl_bbox = resolve_region_cdl_url(
            args.cdl_year,
            args.reference_root,
            "existing local reference-region CDL",
        )
    else:
        cdl_url, cdl_bbox = resolve_region_cdl_url(
            args.cdl_year,
            args.reference_root,
            args.cdl_url,
        )
    print(f"CropScape bbox EPSG:5070: {cdl_bbox}")
    print(f"CropScape download URL: {cdl_url}")
    if cdl_url.startswith("http"):
        download_file(cdl_url, cdl_path)
    else:
        print(f"CDL already exists: {cdl_path}")
    with rasterio.open(cdl_path) as src:
        print(
            f"CDL source: crs={src.crs} shape={src.height}x{src.width} "
            f"dtype={src.dtypes[0]} bounds={tuple(src.bounds)}"
        )
    if args.download_only:
        return

    tile_dirs = sorted(path for path in args.reference_root.iterdir() if path.is_dir())
    if args.tiles:
        selected = set(args.tiles)
        tile_dirs = [path for path in tile_dirs if path.name in selected]
    if not tile_dirs:
        raise RuntimeError("No reference tiles selected")

    jobs: list[tuple[Path, Path]] = []
    for tile_dir in tile_dirs:
        ref_path = reference_b4(tile_dir)
        for year in args.target_years:
            output_path = args.dataset_base / f"{year}_AR" / tile_dir.name / "cdl.tif"
            jobs.append((ref_path, output_path))

    written = 0
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(align_one_mask, cdl_path, ref_path, output_path): output_path
            for ref_path, output_path in jobs
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            output_path = futures[future]
            try:
                written += future.result()
            except Exception as exc:  # noqa: BLE001 - collect every tile failure
                failures.append(f"{output_path}: {exc!r}")
                print(f"[failed] {output_path}: {exc}", flush=True)
            if completed % 50 == 0 or completed == len(jobs):
                print(
                    f"progress={completed}/{len(jobs)} written={written} "
                    f"failures={len(failures)}",
                    flush=True,
                )
    if failures:
        raise SystemExit("CDL alignment failures:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
