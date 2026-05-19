# global_analysis.py (flow-matching version)
from __future__ import annotations

import os
import csv
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import rasterio
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from config import Config
from log import PipelineLogger
from utils.raster_io import (
    read_and_resample_band,
    save_resampled_plot_mask_once,
    ensure_dir,
)
from utils.metrics import calculate_ndvi, calculate_ndwi, calculate_evi, IoU_calculation
from utils.plot_and_csv import _plot_harvest_detection, _write_crop_summary_csv
from harvest_detector import HarvestDetector
from harvest_voter import HarvestVoter


class GlobalAnalyzer:
    """
    Behavior-matching version of run_global_analysis() from esteban_harvest_estimation.py:
      - aligned plot mask saved once
      - loop timestamp folders
      - compute NDVI/NDWI/EVI maps
      - per-crop masking + per-timestamp saves (npy/png)
      - build per-crop time series
      - detect harvest start/end (per index)
      - vote start (with rule) + vote end (fallback)
      - IoU + graph + csv + excel summary
    """

    def __init__(self, cfg: Config, logger: PipelineLogger):
        self.cfg = cfg
        self.logger = logger
        self.detector = HarvestDetector(cfg, logger)
        self.voter = HarvestVoter(cfg, logger)

    def run(
        self,
        *,
        sample_path: Path,
        plot_mask_path: Path,
        output_root: Path,
        year: int,
        run_global: bool = True,
    ) -> None:
        if not run_global:
            return

        sample_path = Path(sample_path)
        plot_mask_path = Path(plot_mask_path)
        output_root = Path(output_root)
        ensure_dir(output_root)

        self.logger.info("🌍 Starting global crop-level analysis...")

        # ---- 1) ensure aligned mask exists (same as original) ----
        aligned_mask_path = sample_path / "plot_mask_resized.npy"
        save_resampled_plot_mask_once(
            plot_mask_path,
            sample_path,
            aligned_mask_path,
        )

        # ---- 2) accumulation containers (same as original) ----
        all_crops: set[int] = set()
        all_timestamps: list[datetime] = []
        data_by_crop_and_time: dict[int, dict[datetime, tuple[float, float, float]]] = {}
        summary_rows: list[dict] = []

        # ---- 3) loop timestamp folders (same as original) ----
        for ts_folder in sorted(os.listdir(sample_path)):
            ts_path = sample_path / ts_folder
            if not ts_path.is_dir():
                continue

            self.logger.info(f"🗂️ Processing timestamp folder: {ts_folder}")

            try:
                files = os.listdir(ts_path)

                # NOTE: original looks for 'B4','B8','B2','B11' substrings
                red_file  = next(f for f in files if "B4" in f)
                nir_file  = next(f for f in files if "B8" in f)
                blue_file = next(f for f in files if "B2" in f)
                swir_file = next(f for f in files if ("B11" in f) or ("B12" in f))

                scl_file = next((f for f in files if "SCL" in f), None)

                with rasterio.open(ts_path / red_file) as red_src:
                    red = red_src.read(1).astype("float32") / 10000.0
                    nir  = read_and_resample_band(ts_path / nir_file,  red_src)
                    blue = read_and_resample_band(ts_path / blue_file, red_src)
                    swir = read_and_resample_band(ts_path / swir_file, red_src)

                    scl = None
                    if scl_file:
                        scl = read_and_resample_band(
                            ts_path / scl_file,
                            red_src,
                            scl=True,
                            resampling=Resampling.nearest,
                        ).astype(np.int16)

                    plot_mask_resized = np.load(aligned_mask_path)
                    plot_mask_resized[red == 0] = 0  # identical behavior

                    # crop labels present in this tile
                    crop_labels = [
                        v for v in np.unique(plot_mask_resized)
                        if (v in self.cfg.valid_crop_labels) and (v != 0)
                    ]

                    # compute index maps
                    ndvi = calculate_ndvi(nir, red)
                    ndwi = calculate_ndwi(nir, swir)
                    evi  = calculate_evi(nir, red, blue)

                    # SCL valid mask (same classes as original)
                    if scl is not None:
                        bad = np.array([3, 8, 9, 10, 11], dtype=np.int16)
                        scl_valid = ~np.isin(scl, bad)
                    else:
                        scl_valid = np.ones_like(red, dtype=bool)

                    timestamp = datetime.strptime(ts_folder, "%Y-%m-%d")
                    all_timestamps.append(timestamp)

                    # ---- 4) loop crops for this timestamp (same as original) ----
                    for crop_label_val in crop_labels:
                        crop_name = self.cfg.crop_dict.get(
                            int(crop_label_val),
                            f"Crop_{crop_label_val}",
                        ).replace("/", "-")

                        if crop_name not in self.cfg.allowed_crops:
                            continue

                        crop_dir = output_root / crop_name
                        ndvi_dir = crop_dir / "NDVI"
                        ndwi_dir = crop_dir / "NDWI"
                        evi_dir  = crop_dir / "EVI"
                        for d in (crop_dir, ndvi_dir, ndwi_dir, evi_dir):
                            ensure_dir(d)

                        mask_crop = (plot_mask_resized == crop_label_val)
                        mask = mask_crop & scl_valid

                        ndvi_val = float(np.nanmean(np.where(mask, ndvi, np.nan)))
                        ndwi_val = float(np.nanmean(np.where(mask, ndwi, np.nan)))
                        evi_val  = float(np.nanmean(np.where(mask, evi,  np.nan)))

                        all_crops.add(int(crop_label_val))
                        data_by_crop_and_time.setdefault(int(crop_label_val), {})[timestamp] = (ndvi_val, ndwi_val, evi_val)

                        # save per-timestamp maps (npy + png) (same logic as original)
                        for index_data, index_val, index_dir, cmap in [
                            (ndvi, ndvi_val, ndvi_dir, "RdYlGn"),
                            (ndwi, ndwi_val, ndwi_dir, "Blues"),
                            (evi,  evi_val,  evi_dir,  "Greens"),
                        ]:
                            masked = np.where(mask, index_data, np.nan)
                            npy_path = index_dir / f"{timestamp:%Y%m%d}.npy"
                            png_path = index_dir / f"{timestamp:%Y%m%d}.png"
                            if (not npy_path.exists()) or (not png_path.exists()):
                                if not np.isnan(index_val):
                                    np.save(npy_path, masked)
                                    plt.imsave(
                                        png_path,
                                        np.nan_to_num(masked, nan=-1),
                                        cmap=cmap, vmin=-1, vmax=1
                                    )
            except Exception as e:
                self.logger.warning(f"❌ Error in {ts_folder}: {e}")
                continue

        # ---- 5) per-crop series -> detect -> vote -> IoU -> graph/csv/excel (same as original) ----
        timestamps_array = sorted(set(all_timestamps))

        for label in sorted(all_crops):
            crop_name = self.cfg.crop_dict.get(label, f"Crop_{label}").replace("/", "-")
            if crop_name not in self.cfg.allowed_crops:
                continue

            crop_dir = output_root / crop_name
            csv_path = crop_dir / f"{crop_name}_summary.csv"
            graph_path = crop_dir / f"{crop_name}_graph.png"

            ndvi_series, ndwi_series, evi_series = [], [], []
            for ts in timestamps_array:
                values = data_by_crop_and_time.get(label, {}).get(ts, (np.nan, np.nan, np.nan))
                ndvi_series.append(values[0])
                ndwi_series.append(values[1])
                evi_series.append(values[2])

            ndvi_array = np.array(ndvi_series, dtype=float)
            ndwi_array = np.array(ndwi_series, dtype=float)
            evi_array  = np.array(evi_series, dtype=float)

            if np.count_nonzero(~np.isnan(ndvi_array)) < self.cfg.min_points:
                continue

            # detection (same call pattern as original)
            detection_ndvi = self.detector.detect(
                timestamps_array,
                ndvi_array,
                label=f"crop:{crop_name}",
                crop_name=crop_name,
            )
            detection_ndwi = self.detector.detect(
                timestamps_array,
                ndwi_array,
                label=f"crop:{crop_name}",
                crop_name=crop_name,
            )
            detection_evi = self.detector.detect(
                timestamps_array,
                evi_array,
                label=f"crop:{crop_name}",
                crop_name=crop_name,
            )

            det_by_src = {
                "NDVI": detection_ndvi,
                "NDWI": detection_ndwi,
                "EVI": detection_evi,
            }

            # start vote (use your voter if it matches, otherwise keep the old rule)
            pred_start, chosen_src, chosen_rule, div_start = self.voter.vote_with_rule(
                {"NDVI": detection_ndvi.start, "NDWI": detection_ndwi.start, "EVI": detection_evi.start},
                {"NDVI": detection_ndvi.start_rule, "NDWI": detection_ndwi.start_rule, "EVI": detection_evi.start_rule},
                kind="start",
                crop_name=crop_name,
                year=year,
            )
            pred_end, div_end = self.voter.vote_simple(
                [detection_ndvi.end, detection_ndwi.end, detection_evi.end],
                timestamps_array,
                kind="end",
                crop_name=crop_name,
                year=year,
            )

            seeding_candidates = [d.seeding_date for d in det_by_src.values() if d.seeding_date is not None]
            seeding_date = None
            if seeding_candidates:
                counts = Counter(seeding_candidates).most_common()
                top_count = counts[0][1]
                top_dates = [d for d, n in counts if n == top_count]
                seeding_date = min(top_dates)

            low_conf_reasons = set()
            for d in det_by_src.values():
                if d.low_conf_reason:
                    low_conf_reasons.update(r for r in d.low_conf_reason.split("|") if r)

            disagreement_days = int(getattr(self.cfg, "seeding_disagreement_days", 14))
            if len(seeding_candidates) >= 2:
                spread = (max(seeding_candidates) - min(seeding_candidates)).days
                if spread > disagreement_days:
                    low_conf_reasons.add("seeding_cross_index_disagreement")
            if seeding_date is None:
                low_conf_reasons.add("missing_seeding_date")

            chosen_det = det_by_src.get(chosen_src)
            rise_date = chosen_det.rise_date if chosen_det else None
            rise_strength = chosen_det.rise_strength if chosen_det else None
            low_confidence = bool(low_conf_reasons)
            low_conf_reason = "|".join(sorted(low_conf_reasons))

            if pred_start is None or pred_end is None:
                continue
            assert pred_start <= pred_end, f"Invalid harvest period for {crop_name}"
            print("crop name:", crop_name)
            iou = IoU_calculation(pred_start, pred_end, self.cfg.gt_windows.get(crop_name))
            self.logger.info(
                f"🏆 {crop_name}: {pred_start.date()} to {pred_end.date()} "
                f"(by {chosen_src}, rule={chosen_rule}), IoU={iou:.3f}"
            )

            # graph 
            if not graph_path.exists():
                _plot_harvest_detection(
                    dates=timestamps_array,
                    ndvi=ndvi_array,
                    ndvi_smooth=detection_ndvi.smoothed,
                    ndwi=ndwi_array,
                    ndwi_smooth=detection_ndwi.smoothed,
                    evi=evi_array,
                    evi_smooth=detection_evi.smoothed,
                    pred_start=pred_start,
                    pred_end=pred_end,
                    crop_name=crop_name,
                    graph_path=graph_path,
                    gt_windows=self.cfg.gt_windows,
                    green_mask=None,
                    logger=self.logger,
                    seeding_date=seeding_date,
                    rise_date=rise_date,
                    low_confidence=low_confidence,
                    low_conf_reason=low_conf_reason,
                )
            else:
                print(f"↪️  Skip plotting {graph_path} (exists)")

            # csv + excel rows (same content style)
            if not csv_path.exists():
                _write_crop_summary_csv(
                    csv_path=csv_path,
                    crop_name=crop_name,
                    timestamps=timestamps_array,
                    ndvi=ndvi_array,
                    ndwi=ndwi_array,
                    evi=evi_array,
                    pred_start=pred_start,
                    pred_end=pred_end,
                    chosen_rule=chosen_rule,
                    div_start=div_start,
                    div_end=div_end,
                    iou=iou,
                    summary_rows=summary_rows,
                    seeding_date=seeding_date,
                    rise_date=rise_date,
                    rise_strength=rise_strength,
                    low_confidence=low_confidence,
                    low_conf_reason=low_conf_reason,
                )

        excel_path = output_root / "harvest_summary_all_crops.xlsx"
        if summary_rows and (not excel_path.exists()):
            pd.DataFrame(summary_rows).to_excel(excel_path, index=False)
            self.logger.info(f"✅ Wrote {excel_path}")
        else:
            self.logger.info(f"↪️  Skip writing {excel_path} (exists or no rows)")
