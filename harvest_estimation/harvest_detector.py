# harvest_detector.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, List, Optional, Tuple

import numpy as np
from scipy.signal import savgol_filter

from config import Config
from log import PipelineLogger


@dataclass
class HarvestDetectionResult:
    start: Optional[datetime]
    end: Optional[datetime]
    smoothed: Optional[np.ndarray]
    start_rule: str
    rise_date: Optional[datetime] = None
    seeding_date: Optional[datetime] = None
    rise_strength: Optional[float] = None
    low_confidence: bool = False
    low_conf_reason: str = ""


class HarvestDetector:
    """
    Detect harvest period from a single index time-series.

    Input:
      - dates: list[datetime] (sorted)
      - series: iterable[float] (same length as dates; may contain NaN)

    Output:
      - HarvestDetectionResult(start, end, smoothed, start_rule)
    """

    def __init__(self, cfg: Config, logger: Optional[PipelineLogger] = None):
        self.cfg = cfg
        self.logger = logger or PipelineLogger(name="HarvestDetector")
        self.default_emergence_days = {
            "Corn": 9,
            "Soybeans": 8,
            "Rice": 7,
            "Cotton": 8,
            "Sorghum": 7,
            "Wheat": 7,
        }

    def _log(self, level: str, msg: str) -> None:
        fn = getattr(self.logger, level, None)
        if callable(fn):
            fn(msg)

    def _first_persistent_rise(
        self,
        deriv: np.ndarray,
        *,
        threshold: float,
        consecutive: int,
    ) -> tuple[Optional[int], list[int]]:
        if consecutive <= 0:
            consecutive = 1
        candidates: list[int] = []
        upper = len(deriv) - consecutive + 1
        for i in range(max(0, upper)):
            window = deriv[i : i + consecutive]
            if np.all(np.isfinite(window)) and np.all(window > threshold):
                candidates.append(i)
        return (candidates[0] if candidates else None), candidates

    def _emergence_days_for_crop(self, crop_name: Optional[str]) -> Optional[int]:
        if not crop_name:
            return None
        cfg_map = getattr(self.cfg, "emergence_days_map", {}) or {}
        if crop_name in cfg_map:
            return int(cfg_map[crop_name])
        return self.default_emergence_days.get(crop_name)

    def _seeding_window_for_crop(self, crop_name: Optional[str], year: int) -> Optional[tuple[datetime, datetime]]:
        if not crop_name:
            return None
        cfg_map = getattr(self.cfg, "seeding_windows", {}) or {}
        window = cfg_map.get(crop_name)
        if not window:
            return None
        try:
            ws, we = window
            return (
                datetime.strptime(f"{year}-{ws}", "%Y-%m-%d"),
                datetime.strptime(f"{year}-{we}", "%Y-%m-%d"),
            )
        except Exception:
            return None

    def detect(
        self,
        dates: List[datetime],
        series: Iterable[float],
        *,
        label: Optional[str] = None,
        crop_name: Optional[str] = None,
    ) -> HarvestDetectionResult:
        min_points = int(self.cfg.min_points)
        prefer_window = int(self.cfg.savgol_window)
        polyorder = int(self.cfg.savgol_polyorder)

        start_rule = "steepest_decline_after_peak"

        y = np.asarray(list(series), dtype=float)
        n = len(y)

        if n != len(dates):
            raise ValueError(f"dates length ({len(dates)}) != series length ({n})")

        valid_mask = np.isfinite(y)
        valid_count = int(np.count_nonzero(valid_mask))

        if n < 3 or valid_count < min_points:
            name = f" ({label})" if label else ""
            self._log("info", f"Skipping{name} – only {valid_count} valid points (<{min_points}).")
            return HarvestDetectionResult(
                None,
                None,
                None,
                "insufficient_data",
                low_confidence=True,
                low_conf_reason="insufficient_data",
            )

        # fill NaNs(can't calculate NDVI/NDWI/EVI) by linear interpolation
        if valid_count < n:
            y = np.interp(np.arange(n), np.flatnonzero(valid_mask), y[valid_mask])

        # choose safe Savitzky–Golay window
        w = min(prefer_window, n)
        if w % 2 == 0:
            w -= 1
        if w <= polyorder:
            w = polyorder + 2
            if w % 2 == 0:
                w += 1
        if w > n:
            w = n if n % 2 == 1 else n - 1

        if w < 3:
            smoothed = y.copy()
        else:
            smoothed = savgol_filter(y, window_length=w, polyorder=min(polyorder, w - 1))

        # Derivative: use numerical gradient
        deriv = np.gradient(smoothed)

        # Peak of growth then find decline
        peak_idx = int(np.argmax(smoothed))
        if peak_idx <= 0 or peak_idx >= n - 1:
            return HarvestDetectionResult(
                None,
                None,
                smoothed,
                "boundary_fallback",
                low_confidence=True,
                low_conf_reason="boundary_peak",
            )

        post_peak_deriv = deriv[peak_idx + 1 :]
        if len(post_peak_deriv) == 0:
            start_idx = min(peak_idx, n - 1)
            start_rule = "no_derivative_after_peak"
        else:
            start_idx = peak_idx + int(np.argmin(post_peak_deriv))

        end_idx = peak_idx + int(np.argmin(smoothed[peak_idx:]))

        # clamp
        start_idx = max(0, min(start_idx, n - 1))
        end_idx = max(0, min(end_idx, n - 1))

        # if start >= end, try a threshold+persistence fallback
        if start_idx >= end_idx and start_idx != peak_idx:
            alpha = 0.85
            k_persist = 3

            def first_persistent_idx(arr, cond, k):
                run = 0
                for i, v in enumerate(arr):
                    run = run + 1 if cond(v) else 0
                    if run >= k:
                        return i - k + 1
                return None

            # after peak and before end
            right_bound = max(peak_idx + 1, min(end_idx, n - 1))
            segment = smoothed[peak_idx + 1 : right_bound]
            threshold = alpha * smoothed[peak_idx]

            under = first_persistent_idx(segment, lambda v: v <= threshold, k_persist)
            if under is not None:
                start_idx = (peak_idx + 1) + under
                start_rule = f"threshold_{alpha}_p{k_persist}"
            else:
                post_deriv = np.gradient(smoothed)[peak_idx + 1 : right_bound]
                start_idx = (peak_idx + 1) + int(np.argmin(post_deriv)) if len(post_deriv) else peak_idx
                start_rule = "steepest_decline_after_peak_retry" if len(post_deriv) else "no_derivative_after_peak_retry"

            # ensure start < end
            start_idx = max(peak_idx + 1, min(start_idx, end_idx - 1))

        rise_threshold = float(getattr(self.cfg, "seeding_rise_threshold", 0.02))
        rise_consecutive = int(getattr(self.cfg, "seeding_rise_consecutive", 2))
        edge_buffer = int(getattr(self.cfg, "seeding_edge_buffer", 2))
        ambiguity_gap = int(getattr(self.cfg, "seeding_ambiguity_gap", 2))
        season_year = dates[start_idx].year if start_idx < len(dates) else (dates[0].year if dates else 2019)

        rise_idx, rise_candidates = self._first_persistent_rise(
            deriv,
            threshold=rise_threshold,
            consecutive=rise_consecutive,
        )

        low_conf_reasons: list[str] = []
        rise_strength: Optional[float] = None
        rise_date: Optional[datetime] = None
        seeding_date: Optional[datetime] = None

        if rise_idx is None:
            low_conf_reasons.append("no_persistent_rise")
            if np.all(~np.isfinite(deriv)):
                rise_idx = None
            else:
                rise_idx = int(np.nanargmax(deriv))
                low_conf_reasons.append("fallback_max_rise")

        if rise_idx is not None:
            rise_idx = max(0, min(rise_idx, n - 1))
            rise_date = dates[rise_idx]
            rise_strength = float(deriv[rise_idx]) if np.isfinite(deriv[rise_idx]) else None

            if rise_idx <= edge_buffer or rise_idx >= (n - 1 - edge_buffer):
                low_conf_reasons.append("rise_at_series_edge")

        if len(rise_candidates) >= 2 and (rise_candidates[1] - rise_candidates[0]) <= ambiguity_gap:
            low_conf_reasons.append("ambiguous_rise_candidates")

        emergence_days = self._emergence_days_for_crop(crop_name)
        if rise_date is None:
            low_conf_reasons.append("missing_rise_date")
        elif emergence_days is None:
            low_conf_reasons.append("missing_emergence_days")
        else:
            seeding_date = rise_date - timedelta(days=int(emergence_days))
            seeding_window = self._seeding_window_for_crop(crop_name, season_year)
            if seeding_window is not None:
                ws, we = seeding_window
                if seeding_date < ws or seeding_date > we:
                    low_conf_reasons.append("seeding_out_of_window")

        return HarvestDetectionResult(
            dates[start_idx],
            dates[end_idx],
            smoothed,
            start_rule,
            rise_date=rise_date,
            seeding_date=seeding_date,
            rise_strength=rise_strength,
            low_confidence=bool(low_conf_reasons),
            low_conf_reason="|".join(sorted(set(low_conf_reasons))),
        )
