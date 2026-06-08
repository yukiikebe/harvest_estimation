from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from doy_prediction.tile_cnn_data import (
    TileCropDataset,
    TileCropRecord,
    build_tile_crop_records,
    get_feature_names,
    load_crop_windows_yaml,
    observations_to_tensor,
)
from doy_prediction.metrics import harvest_window_iou, maybe_harvest_window_iou
from doy_prediction.train_tile_cnn import evaluate_model
from doy_prediction.tile_cnn_model import TileCNNRegressor, normalized_to_doy
from doy_prediction.tile_rnn_model import TileRNNRegressor


class TileCnnDataTests(unittest.TestCase):
    def test_build_records_multiple_crops(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "outputs" / "2020_AR" / "0_1"
            base.mkdir(parents=True)
            write_test_workbook(
                base / "harvest_summary_all_crops.xlsx",
                rows=[
                    ["Corn", "2020-01-05", 0.10, "", "", "", "", "", "", ""],
                    ["Corn", "2020-01-10", 0.20, "", "", "", "", "", "", ""],
                    ["Corn", "2020-01-15", 0.30, "", "", "Start", "", "", "", ""],
                    ["Corn", "2020-12-31", 0.40, "", "", "End", "", "", "", ""],
                    ["Rice", "2020-02-01", 0.15, 0.01, 0.03, "", "", "", "", ""],
                    ["Rice", "2020-02-10", "", 0.02, 0.04, "Start", "", "", "", ""],
                    ["Rice", "2020-11-20", 0.55, 0.11, 0.30, "End", "", "", "", ""],
                ],
            )

            records = build_tile_crop_records(
                base.parent.parent,
                crops=["Corn", "Rice"],
                years=[2020],
                feature_set="all_indices",
                min_points=2,
            )

            self.assertEqual(len(records), 2)
            corn = next(record for record in records if record.crop == "Corn")
            rice = next(record for record in records if record.crop == "Rice")
            self.assertEqual(corn.start_doy, 15)
            self.assertEqual(corn.end_doy, 366)
            self.assertEqual(rice.x.shape, (4, 73))

    def test_feature_options_and_filtering(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "outputs" / "2021_AR" / "0_2"
            base.mkdir(parents=True)
            write_test_workbook(
                base / "harvest_summary_all_crops.xlsx",
                rows=[
                    ["Rice", "2021-01-05", "", 0.10, 0.20, "", "", "", "", ""],
                    ["Rice", "2021-01-15", 0.30, "", "", "Start", "", "", "", ""],
                    ["Rice", "2021-11-20", 0.40, 0.20, 0.10, "End", "", "", "", ""],
                ],
            )

            ndvi_records = build_tile_crop_records(
                base.parent.parent,
                crops=["Rice"],
                years=[2021],
                feature_set="ndvi_only",
                min_points=2,
            )
            all_records = build_tile_crop_records(
                base.parent.parent,
                crops=["Rice"],
                years=[2021],
                feature_set="all_indices",
                min_points=2,
            )

            self.assertEqual(len(ndvi_records), 1)
            self.assertEqual(ndvi_records[0].x.shape, (2, 73))
            self.assertEqual(all_records[0].x.shape, (4, 73))

    def test_observations_to_tensor_fixed_length(self) -> None:
        tensor = observations_to_tensor(
            [
                (5, {"NDVI": 0.2, "NDWI": 0.1, "EVI": np.nan}),
                (33, {"NDVI": 0.5, "NDWI": np.nan, "EVI": 0.3}),
            ],
            feature_names=get_feature_names("all_indices"),
        )
        self.assertEqual(tensor.shape, (4, 73))
        self.assertGreater(tensor[-1, 0], 0.0)
        self.assertEqual(float(tensor[-1, 10]), 0.0)

    def test_crop_window_filters_observations_but_keeps_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "outputs" / "2021_AR" / "0_3"
            base.mkdir(parents=True)
            write_test_workbook(
                base / "harvest_summary_all_crops.xlsx",
                rows=[
                    ["Rice", "2021-06-20", 0.10, 0.01, 0.03, "", "", "", "", ""],
                    ["Rice", "2021-07-20", 0.15, 0.02, 0.04, "", "", "", "", ""],
                    ["Rice", "2021-08-15", 0.30, 0.03, 0.05, "Start", "", "", "", ""],
                    ["Rice", "2021-11-20", 0.55, 0.11, 0.30, "End", "", "", "", ""],
                ],
            )

            records = build_tile_crop_records(
                base.parent.parent,
                crops=["Rice"],
                years=[2021],
                feature_set="all_indices",
                min_points=2,
                crop_windows={"Rice": ("07-01", "09-30")},
            )

            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record.start_doy, 227)
            self.assertEqual(record.end_doy, 324)
            self.assertEqual(record.num_observations, 2)

    def test_crop_window_drops_record_with_insufficient_points(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "outputs" / "2021_AR" / "0_4"
            base.mkdir(parents=True)
            write_test_workbook(
                base / "harvest_summary_all_crops.xlsx",
                rows=[
                    ["Rice", "2021-06-20", 0.10, 0.01, 0.03, "", "", "", "", ""],
                    ["Rice", "2021-08-15", 0.30, 0.03, 0.05, "Start", "", "", "", ""],
                    ["Rice", "2021-11-20", 0.55, 0.11, 0.30, "End", "", "", "", ""],
                ],
            )

            records = build_tile_crop_records(
                base.parent.parent,
                crops=["Rice"],
                years=[2021],
                feature_set="all_indices",
                min_points=2,
                crop_windows={"Rice": ("07-01", "09-30")},
            )

            self.assertEqual(records, [])

    def test_load_crop_windows_yaml_supports_list_and_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "windows.yaml"
            path.write_text(
                """Rice:
  - '07-14'
  - '12-31'
Corn:
  start: '06-25'
  end: '12-31'
""",
                encoding="utf-8",
            )

            windows = load_crop_windows_yaml(path)

            self.assertEqual(windows["Rice"], ("07-14", "12-31"))
            self.assertEqual(windows["Corn"], ("06-25", "12-31"))

    def test_model_forward_and_postprocess(self) -> None:
        model = TileCNNRegressor(in_channels=4)
        x = torch.randn(3, 4, 73)
        out = model(x)
        self.assertEqual(tuple(out.shape), (3, 2))
        doy = normalized_to_doy(out)
        self.assertEqual(doy.shape, (3, 2))
        self.assertTrue(np.all(doy[:, 0] <= doy[:, 1]))

    def test_rnn_model_forward_reads_fixed_length_input(self) -> None:
        model = TileRNNRegressor(in_channels=4, hidden_size=8, num_layers=1)
        x = torch.randn(3, 4, 73)

        out = model(x)

        self.assertEqual(tuple(out.shape), (3, 2))

    def test_rnn_model_forward_supports_per_layer_hidden_sizes(self) -> None:
        model = TileRNNRegressor(in_channels=4, hidden_sizes=(8, 16, 32))
        x = torch.randn(3, 4, 73)

        out = model(x)

        self.assertEqual(tuple(out.shape), (3, 2))
        self.assertEqual(len(model.rnn_layers), 3)
        self.assertEqual(model.rnn_layers[0].hidden_size, 8)
        self.assertEqual(model.rnn_layers[1].hidden_size, 16)
        self.assertEqual(model.rnn_layers[2].hidden_size, 32)

    def test_harvest_window_iou(self) -> None:
        self.assertAlmostEqual(harvest_window_iou(60, 110, 50, 100), 40.0 / 60.0)
        self.assertIsNone(
            maybe_harvest_window_iou(
                pred_start_doy=60,
                pred_end_doy=110,
                true_start_doy=None,
                true_end_doy=100,
            )
        )

    def test_evaluate_model_reports_iou(self) -> None:
        record = TileCropRecord(
            year=2020,
            tile="0_1",
            crop="Rice",
            x=np.zeros((4, 73), dtype=np.float32),
            start_doy=50,
            end_doy=100,
            source_workbook="dummy.xlsx",
            feature_set="all_indices",
            num_observations=2,
        )
        loader = DataLoader(TileCropDataset([record]), batch_size=1, shuffle=False)
        model = ConstantPredictionModel(start_doy=60, end_doy=110)

        metrics = evaluate_model(model, loader, torch.device("cpu"))

        self.assertAlmostEqual(metrics["mae_start"], 10.0)
        self.assertAlmostEqual(metrics["mae_end"], 10.0)
        self.assertAlmostEqual(metrics["iou_mean"], 40.0 / 60.0)
        self.assertAlmostEqual(metrics["ordered_pct"], 1.0)


def write_test_workbook(path: Path, rows: list[list[object]]) -> None:
    headers = ["Crop", "Date", "NDVI", "NDWI", "EVI", "Harvest", "start_rule", "div_start", "div_end", "IoU"]
    worksheet_rows = [headers, *rows]

    row_xml = []
    for row_idx, row in enumerate(worksheet_rows, start=1):
        cells = []
        for col_idx, value in enumerate(row, start=1):
            ref = f"{column_name(col_idx)}{row_idx}"
            if value == "":
                cells.append(f'<c r="{ref}" t="inlineStr"></c>')
                continue
            if isinstance(value, (int, float)):
                cells.append(f'<c r="{ref}" t="n"><v>{value}</v></c>')
            else:
                cells.append(
                    f'<c r="{ref}" t="inlineStr"><is><t>{value}</t></is></c>'
                )
        row_xml.append(f'<row r="{row_idx}">{"".join(cells)}</row>')

    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetData>'
        f'{"".join(row_xml)}'
        '</sheetData>'
        '</worksheet>'
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
            '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
            '</Types>',
        )
        zf.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
            '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
            '</Relationships>',
        )
        zf.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets>'
            '</workbook>',
        )
        zf.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
            '</Relationships>',
        )
        zf.writestr(
            "docProps/core.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" />',
        )
        zf.writestr(
            "docProps/app.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" />',
        )
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)


def column_name(index: int) -> str:
    out = []
    while index:
        index, rem = divmod(index - 1, 26)
        out.append(chr(ord("A") + rem))
    return "".join(reversed(out))


class ConstantPredictionModel(nn.Module):
    def __init__(self, *, start_doy: int, end_doy: int) -> None:
        super().__init__()
        self.output = torch.tensor([[start_doy / 366.0, end_doy / 366.0]], dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output.repeat(x.shape[0], 1).to(x.device)


if __name__ == "__main__":
    unittest.main()
