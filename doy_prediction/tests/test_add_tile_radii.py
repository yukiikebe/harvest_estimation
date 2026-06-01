from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from create_doy_prediction_input.add_tile_radii import (
    RADIUS_FIELDS,
    augment_rows_with_radii,
    compute_radii,
    write_rows,
)


class AddTileRadiiTests(unittest.TestCase):
    def test_compute_radii_uses_tile_sizes(self) -> None:
        row = {
            "tile_size_x_crs": "13360.0",
            "tile_size_y_crs": "12280.0",
            "bounds_left": "0.0",
            "bounds_right": "13360.0",
            "bounds_bottom": "0.0",
            "bounds_top": "12280.0",
        }

        radii = compute_radii(row)

        self.assertEqual(radii["radius_x_crs"], 6680.0)
        self.assertEqual(radii["radius_y_crs"], 6140.0)
        self.assertAlmostEqual(radii["radius_corner_crs"], 9073.147193780116)

    def test_compute_radii_falls_back_to_bounds(self) -> None:
        row = {
            "bounds_left": "-92.0",
            "bounds_right": "-90.6",
            "bounds_bottom": "34.4",
            "bounds_top": "35.0",
        }

        radii = compute_radii(row)

        self.assertAlmostEqual(radii["radius_x_crs"], 0.7)
        self.assertAlmostEqual(radii["radius_y_crs"], 0.3)
        self.assertAlmostEqual(radii["radius_corner_crs"], 0.7615773105863908)

    def test_write_rows_appends_radius_columns(self) -> None:
        rows = augment_rows_with_radii(
            [
                {
                    "tile": "0_0",
                    "tile_size_x_crs": "4.0",
                    "tile_size_y_crs": "6.0",
                    "bounds_left": "-91.0",
                    "bounds_right": "-87.0",
                    "bounds_bottom": "34.0",
                    "bounds_top": "40.0",
                }
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_csv = Path(tmpdir) / "with_radii.csv"
            fieldnames = [
                "tile",
                "tile_size_x_crs",
                "tile_size_y_crs",
                "bounds_left",
                "bounds_right",
                "bounds_bottom",
                "bounds_top",
            ] + RADIUS_FIELDS
            write_rows(out_csv, rows, fieldnames)

            with out_csv.open("r", newline="", encoding="utf-8") as handle:
                loaded_rows = list(csv.DictReader(handle))

        self.assertEqual(loaded_rows[0]["radius_x_crs"], "2.0")
        self.assertEqual(loaded_rows[0]["radius_y_crs"], "3.0")
        self.assertEqual(loaded_rows[0]["radius_corner_crs"], "3.605551275463989")


if __name__ == "__main__":
    unittest.main()
