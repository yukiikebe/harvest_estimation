from __future__ import annotations


def harvest_window_iou(
    pred_start_doy: int,
    pred_end_doy: int,
    true_start_doy: int,
    true_end_doy: int,
) -> float:
    intersection = max(0, min(pred_end_doy, true_end_doy) - max(pred_start_doy, true_start_doy))
    union = max(pred_end_doy, true_end_doy) - min(pred_start_doy, true_start_doy)
    if union <= 0:
        return 0.0
    return float(intersection / union)


def maybe_harvest_window_iou(
    *,
    pred_start_doy: int,
    pred_end_doy: int,
    true_start_doy: int | None,
    true_end_doy: int | None,
) -> float | None:
    if true_start_doy is None or true_end_doy is None:
        return None
    return harvest_window_iou(pred_start_doy, pred_end_doy, true_start_doy, true_end_doy)
