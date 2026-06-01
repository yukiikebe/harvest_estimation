from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from create_doy_prediction_input.export_tile_centers import (
    FIELDNAMES,
    build_tile_center_rows,
    get_raster_center_lon_lat,
    resolve_tile_raster,
    write_tile_center_csv,
)


class ExportTileCentersTests(unittest.TestCase):
    def test_get_raster_center_lon_lat_wgs84(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            raster_path = Path(tmpdir) / "B4_2020-01-01.tif"
            self._write_tif(
                raster_path,
                crs="EPSG:4326",
                transform=from_origin(-91.0, 35.0, 0.1, 0.1),
                width=4,
                height=6,
            )

            lon, lat, crs = get_raster_center_lon_lat(raster_path)

            self.assertEqual(crs, "EPSG:4326")
            self.assertAlmostEqual(lon, -90.8)
            self.assertAlmostEqual(lat, 34.7)

    def test_resolve_tile_raster_prefers_first_b4_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tile_dir = Path(tmpdir) / "0_0"
            ts_late = tile_dir / "2020-02-01"
            ts_early = tile_dir / "2020-01-01"
            ts_late.mkdir(parents=True)
            ts_early.mkdir(parents=True)
            self._write_tif(ts_late / "B4_2020-02-01.tif", crs="EPSG:4326", transform=from_origin(-90.0, 35.0, 0.1, 0.1), width=2, height=2)
            self._write_tif(ts_early / "B4_2020-01-01.tif", crs="EPSG:4326", transform=from_origin(-91.0, 36.0, 0.1, 0.1), width=2, height=2)

            raster_path = resolve_tile_raster(tile_dir, raster_name=None)

            self.assertEqual(raster_path.name, "B4_2020-01-01.tif")
            self.assertEqual(raster_path.parent.name, "2020-01-01")

    def test_build_tile_center_rows_uses_b4_by_default_and_includes_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "dataset"
            for tile_name, west, north in [
                ("1_0", -91.0, 35.0),
                ("0_2", -92.0, 36.0),
            ]:
                tile_dir = dataset_root / tile_name / "2020-01-01"
                tile_dir.mkdir(parents=True)
                self._write_tif(
                    tile_dir / "B4_2020-01-01.tif",
                    crs="EPSG:4326",
                    transform=from_origin(west, north, 0.1, 0.1),
                    width=2,
                    height=2,
                )

            rows = build_tile_center_rows(dataset_root)
            self.assertEqual([row["tile"] for row in rows], ["0_2", "1_0"])
            self.assertTrue(rows[0]["source_raster"].endswith("2020-01-01/B4_2020-01-01.tif"))
            self.assertEqual(rows[0]["width_pixels"], 2)
            self.assertEqual(rows[0]["height_pixels"], 2)
            self.assertAlmostEqual(rows[0]["pixel_size_x"], 0.1)
            self.assertAlmostEqual(rows[0]["pixel_size_y"], 0.1)
            self.assertAlmostEqual(rows[0]["tile_size_x_crs"], 0.2)
            self.assertAlmostEqual(rows[0]["tile_size_y_crs"], 0.2)
            self.assertAlmostEqual(rows[0]["bounds_left"], -92.0)
            self.assertAlmostEqual(rows[0]["bounds_top"], 36.0)
            self.assertAlmostEqual(rows[0]["transform_a"], 0.1)
            self.assertAlmostEqual(rows[0]["transform_e"], -0.1)

            filtered_rows = build_tile_center_rows(dataset_root, tiles=["1_0"])
            self.assertEqual(len(filtered_rows), 1)
            self.assertEqual(filtered_rows[0]["tile"], "1_0")

    def test_write_tile_center_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_csv = Path(tmpdir) / "tile_centers.csv"
            rows = [
                {
                    "tile": "0_0",
                    "center_latitude": 34.7,
                    "center_longitude": -90.8,
                    "center_x_crs": -90.8,
                    "center_y_crs": 34.7,
                    "width_pixels": 4,
                    "height_pixels": 6,
                    "pixel_size_x": 0.1,
                    "pixel_size_y": 0.1,
                    "tile_size_x_crs": 0.4,
                    "tile_size_y_crs": 0.6,
                    "bounds_left": -91.0,
                    "bounds_bottom": 34.4,
                    "bounds_right": -90.6,
                    "bounds_top": 35.0,
                    "transform_a": 0.1,
                    "transform_b": 0.0,
                    "transform_c": -91.0,
                    "transform_d": 0.0,
                    "transform_e": -0.1,
                    "transform_f": 35.0,
                    "source_crs": "EPSG:4326",
                    "source_raster": "/tmp/0_0/2020-01-01/B4_2020-01-01.tif",
                }
            ]

            write_tile_center_csv(out_csv, rows)

            with out_csv.open("r", encoding="utf-8", newline="") as handle:
                loaded_rows = list(csv.DictReader(handle))

            self.assertEqual(FIELDNAMES, list(loaded_rows[0].keys()))
            self.assertEqual(len(loaded_rows), 1)
            self.assertEqual(loaded_rows[0]["tile"], "0_0")
            self.assertEqual(loaded_rows[0]["source_crs"], "EPSG:4326")
            self.assertEqual(loaded_rows[0]["width_pixels"], "4")

    def test_explicit_raster_name_can_use_cdl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "dataset"
            tile_dir = dataset_root / "0_0"
            tile_dir.mkdir(parents=True)
            self._write_tif(
                tile_dir / "cdl.tif",
                crs="EPSG:4326",
                transform=from_origin(-91.0, 35.0, 30.0, 30.0),
                width=2,
                height=2,
            )

            rows = build_tile_center_rows(dataset_root, raster_name="cdl.tif")

            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0]["source_raster"].endswith("0_0/cdl.tif"))
            self.assertEqual(rows[0]["pixel_size_x"], 30.0)

    def test_missing_tile_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "dataset"
            dataset_root.mkdir()
            tile_dir = dataset_root / "0_0" / "2020-01-01"
            tile_dir.mkdir(parents=True)
            self._write_tif(
                tile_dir / "B4_2020-01-01.tif",
                crs="EPSG:4326",
                transform=from_origin(-91.0, 35.0, 0.1, 0.1),
                width=2,
                height=2,
            )

            with self.assertRaises(FileNotFoundError):
                build_tile_center_rows(dataset_root, tiles=["9_9"])

    def test_missing_raster_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "dataset"
            tile_dir = dataset_root / "0_0"
            tile_dir.mkdir(parents=True)

            with self.assertRaises(FileNotFoundError):
                build_tile_center_rows(dataset_root)

    def test_missing_crs_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "dataset"
            tile_dir = dataset_root / "0_0" / "2020-01-01"
            tile_dir.mkdir(parents=True)
            self._write_tif(
                tile_dir / "B4_2020-01-01.tif",
                crs=None,
                transform=from_origin(-91.0, 35.0, 0.1, 0.1),
                width=2,
                height=2,
            )

            with self.assertRaises(ValueError):
                build_tile_center_rows(dataset_root)

    def _write_tif(
        self,
        path: Path,
        *,
        crs: str | None,
        transform,
        width: int,
        height: int,
    ) -> None:
        data = np.ones((height, width), dtype=np.uint8)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=data.dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(data, 1)


if __name__ == "__main__":
    unittest.main()
