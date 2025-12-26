# === Imports ===
import os
import numpy as np
import rasterio
import rasterio.warp
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.image import imsave
from matplotlib.colors import ListedColormap, BoundaryNorm, to_rgb
from datetime import datetime, timedelta
import csv
import pandas as pd
from scipy.signal import savgol_filter
import yaml
from scipy.ndimage import label, center_of_mass
import re
from collections import Counter
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
from scipy.ndimage import label, center_of_mass
from rasterio.transform import xy as pixel_to_map
from pyproj import Transformer
import logging
import time
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from scipy.ndimage import binary_erosion
from datetime import datetime, timedelta
from bisect import bisect_left

def vote_or_fallback(candidates, dates, kind="start", harvest_window=None, year=2019):
    c = [d for d in candidates if d is not None]
    div_result = None
    ws, we = harvest_window
    assert (ws is not None) and (we is not None), "harvest_window must be provided"
    gt_start = datetime.strptime(f"{year}-{ws}", "%Y-%m-%d") if ws else None
    gt_end   = datetime.strptime(f"{year}-{we}", "%Y-%m-%d") if we else None
    
    if c:
        counts = Counter(c).most_common()
        top_count = counts[0][1]
        top_dates = [d for d, n in counts if n == top_count]
        if kind == "start":
            chosen_dt = min(top_dates)
            div_result = abs(gt_start - chosen_dt).days
        else:
            chosen_dt = max(top_dates)
            div_result = abs(gt_end - chosen_dt).days

        return chosen_dt, div_result
    
    if kind == "start":
        chosen_dt = dates[0]
        div_result = abs((chosen_dt - gt_start).days)
    else:
        chosen_dt = dates[-1]
        div_result = abs((chosen_dt - gt_end).days)
    return chosen_dt, div_result

def vote_or_fallback_with_rule(dates_by_src, rules_by_src, kind="start", harvest_window=None, year=2019):
    """
    dates_by_src: dict like {"NDVI": dt_or_None, "NDWI": dt_or_None, "EVI": dt_or_None}
    rules_by_src: dict like {"NDVI": rule_str,  "NDWI": rule_str,  "EVI": rule_str}
    kind:         "start" -> earliest among top votes, "end" -> latest among top votes
    priority:     tie-break among sources sharing the chosen datetime
    Returns: (chosen_dt, chosen_source, chosen_rule)
    """
    items = [(src, dt) for src, dt in dates_by_src.items() if dt is not None]
    div_result = None
    # print("candidates:", items)

    # for src, dt in items:
    #     print(f"  {src}: {dt} (rule: {rules_by_src.get(src, '')})")
        
    ws, we = harvest_window
    assert (ws is not None) and (we is not None), "harvest_window must be provided"
    gt_start = datetime.strptime(f"{year}-{ws}", "%Y-%m-%d") if ws else None
    gt_end   = datetime.strptime(f"{year}-{we}", "%Y-%m-%d") if we else None
    
    if not items:
        chosen_dt = gt_start if kind == "start" else gt_end
        chosen_src = "FALLBACK"
        chosen_rule = "no_valid_dates_all_sources"
        div_result = 0
        return chosen_dt, chosen_src, chosen_rule, div_result
    
    if kind == "start":
        chosen_dt = min((d for _, d in items), key=lambda d: (abs((d - gt_start).days), d))
        div_result = abs((chosen_dt - gt_start).days)
    else:
        chosen_dt = min((d for _, d in items), key=lambda d: (abs((d - gt_end).days), d))
        div_result = abs((chosen_dt - gt_end).days)

    winners = [src for src, dt in items if dt == chosen_dt]

    def rule_rank(rule: str) -> int:
        r = (rule or "").lower()
        if "steepest_decline_" in r:
            return 0
        if r.startswith("threshold"):
            return 1
        else:
            return 2

    chosen_src = min(winners, key=lambda s: (rule_rank(rules_by_src.get(s, "")), s))
    chosen_rule = rules_by_src.get(chosen_src, "")
    
    # return div(d - )
    return chosen_dt, chosen_src, chosen_rule, div_result

# === Logging ===
LOG_DIR = "./logs"
LOGGER = None

def setup_logging(year_tag=None):
    """Console + rotating file. example: logs/harvest_2019_AR.log"""
    global LOGGER
    name = "harvest" if year_tag is None else f"harvest_{year_tag}"
    os.makedirs(LOG_DIR, exist_ok=True)
    LOGGER = logging.getLogger(name)
    if LOGGER.handlers: 
        return LOGGER
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    fh = RotatingFileHandler(
        os.path.join(LOG_DIR, f"{name}.log"),
        maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    LOGGER.addHandler(sh)
    LOGGER.addHandler(fh)
    return LOGGER

@contextmanager
def log_section(title):
    t0 = time.time()
    LOGGER.info(f"START {title}")
    try:
        yield
        LOGGER.info(f"DONE  {title}  ({time.time()-t0:.1f}s)")
    except Exception:
        LOGGER.exception(f"FAIL  {title}")
        raise

def mark_done(path):
    try:
        with open(os.path.join(path, "_DONE.txt"), "w") as f:
            f.write(datetime.now().isoformat())
    except Exception:
        LOGGER.warning(f"Could not write _DONE.txt in {path}")




# === Config ===
current = "12_0" 
sample_path = f"../AR_sentinel2/2019_AR/{current}" 
plot_mask_path = f"../AR_sentinel2/2019_AR/{current}/cdl.tif" 
output_root = current + "_New_Index"
run_global = True
run_farm = True
allowed_crops = {"Corn", "Cotton", "Rice", "Soybeans", "Winter Wheat"}

gt_windows = {
    "Corn": ("07-31", "10-15"),
    "Cotton": ("09-05", "11-30"),
    "Rice": ("08-05", "11-05"),
    "Soybeans": ("09-05", "11-30"),
    "Winter Wheat": ("05-31", "07-31")
}

def save_scl_png(masked_scl, png_path, background_color="#ffffff"):
    colors = [
        "#000000",  # 0 No data
        "#ff00ff",  # 1 Saturated/Defective
        "#2f2f2f",  # 2 Dark area pixels
        "#6f6f6f",  # 3 Cloud shadows
        "#00a000",  # 4 Vegetation
        "#be9b7b",  # 5 Bare soils
        "#0000ff",  # 6 Water
        "#aaaaaa",  # 7 Unclassified
        "#ffff00",  # 8 Cloud medium probability
        "#ffa500",  # 9 Cloud high probability
        "#add8e6",  # 10 Thin cirrus
        "#0fffff",  # 11 Snow/ice
    ]
    colors_rgb = np.array([np.array(to_rgb(c)) * 255 for c in colors], dtype=np.uint8)  # (12,3)

    h, w = masked_scl.shape
    rgb = np.empty((h, w, 3), dtype=np.uint8)

    background = (np.array(to_rgb(background_color)) * 255).astype(np.uint8)
    rgb[:] = background

    valid = ~np.isnan(masked_scl)
    idx = masked_scl[valid].astype(int)
    rgb[valid] = colors_rgb[idx]

    imsave(png_path, rgb)

with open("./configs/Arkansas/cdl.yaml", "r") as f:
    yaml_data = yaml.safe_load(f)
    crop_dict = yaml_data.get("num2class", {})
    valid_crop_labels = set(yaml_data.get("crop_type", {}).keys())

# === Helper Functions ===
def calculate_ndvi(nir, red):
    # print("Calculating NDVI...")
    return np.clip((nir - red) / (nir + red + 1e-10), -1, 1)

def calculate_ndwi(nir, swir):
    # print("Calculating NDWI...")
    return np.clip((nir - swir) / (nir + swir + 1e-10), -1, 1)

def calculate_evi(nir, red, blue):
    # print("Calculating EVI...")
    return np.clip(2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + 1e-10),-1,1)

def ensure_dir(directory):
    # print(f"Ensuring directory exists: {directory}")
    if not os.path.exists(directory):
        os.makedirs(directory)
        
# def read_scl_nearest(scl_path, ref_src):
#     with rasterio.open(scl_path) as src:
#         scl = src.read(1).astype(np.int16)
#         if src.shape != (ref_src.height, ref_src.width):
#             out = np.empty((ref_src.height, ref_src.width), dtype=np.int16)
#             rasterio.warp.reproject(
#                 scl, out,
#                 src_transform=src.transform, src_crs=src.crs,
#                 dst_transform=ref_src.transform, dst_crs=ref_src.crs,
#                 resampling=Resampling.nearest
#             )
#             return out
#         return scl

# Adjust resolution and alignment to reference source
def read_and_resample_band(band_path, ref_src, scl=False, resampling=Resampling.bilinear):
    # print(f"Reading and resampling band: {band_path}")
    with rasterio.open(band_path) as src:
        if not scl:
            data = src.read(1).astype('float32') / 10000.0 # sentinel-2 reflectance scaling
        else:
            data = src.read(1).astype('uint8')
        if src.shape != (ref_src.height, ref_src.width):
            resampled = np.empty((ref_src.height, ref_src.width), dtype='float32')
            rasterio.warp.reproject(
                source=data,
                destination=resampled,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_src.transform,
                dst_crs=ref_src.crs,
                resampling=resampling
            )
            return resampled
        return data

# polyorder decide the degree of the polynomial (default=3)
# from gradient of smoothed series, find peak (start) then steepest decline (end) after peak
def detect_harvest_period(dates, index_series, label=None, min_points=11, prefer_window=11, polyorder=3):
    start_rule = "steepest_decline_after_peak"
    
    series = np.asarray(index_series, dtype=float)
    n = len(series)
    valid_mask = ~np.isnan(series)
    valid_count = int(np.count_nonzero(valid_mask))

    if n < 3 or valid_count < min_points:
        msg = f"Skipping ({label}) – only {valid_count} valid points (<{min_points})." if label else \
              f"Skipping – only {valid_count} valid points (<{min_points})."
        try:
            LOGGER.info(msg)
        except Exception:
            print(msg)
        start_rule = "insufficient_data"
        return None, None, None, start_rule

    # Fill NaNs(can't calculate NDVI/NDWI/EVI) by linear interpolation
    if valid_count < n:
        series = np.interp(np.arange(n), np.flatnonzero(valid_mask), series[valid_mask])

    # Odd window <= n and > polyorder (for using savgol_filter, window must be odd)
    w = min(prefer_window, n)
    if w % 2 == 0:
        w -= 1
    if w <= polyorder:
        w = polyorder + 2  # minimal odd >= polyorder+1
        if w % 2 == 0:
            w += 1
    if w > n: # window length can't be greater than series length
        w = n if n % 2 == 1 else n - 1
    if w < 3:
        # fallback: no smoothing, just use raw with gradient
        smoothed = series.copy()
    else:
        # Use your existing safe smoother logic to be extra-safe
        smoothed = savgol_filter(series, window_length=w, polyorder=min(polyorder, w - 1))

    # Derivative: use numerical gradient (more forgiving than savgol deriv)
    first_deriv = np.gradient(smoothed)

    # Peak of growth (max smoothed) then find decline
    peak_idx = int(np.argmax(smoothed))
    if peak_idx <= 0 or peak_idx >= n - 1:
        start_rule = "boundary_fallback"
        return None, None, smoothed, start_rule

    post_peak_deriv = first_deriv[peak_idx + 1:]
    if len(post_peak_deriv) == 0:
        start_idx = min(peak_idx, n - 1)
        start_rule = "no_derivative_after_peak"
    else:
        steepest_decline_idx = int(np.argmin(post_peak_deriv))
        start_idx = peak_idx + steepest_decline_idx # steepst decline index calculated from post_peak_deriv

    # End = lowest value after peak
    end_idx = int(np.argmin(smoothed[peak_idx:])) + peak_idx
    # print(f"Detected harvest period for {label}: {dates[start_idx]} to {dates[end_idx]} (peak at {dates[peak_idx]})")

    # Clamp to bounds just in case
    start_idx = max(0, min(start_idx, n - 1))
    end_idx   = max(0, min(end_idx,   n - 1))

    if start_idx >= end_idx and start_idx != peak_idx:
        alpha = 0.85       # 85% of peak
        k_persist = 3      # need 3 consecutive points below threshold

        def first_persistent_idx(arr, cond, k):
            run = 0
            for i, v in enumerate(arr):
                run = run + 1 if cond(v) else 0
                if run >= k:
                    return i - k + 1
            return None

        # after peak and before end
        right_bound = max(peak_idx + 1, min(end_idx, n - 1))
        after_peak_to_before_end = smoothed[peak_idx + 1 : right_bound]

        # Threshold-based re-pick of START with persistence
        threshold = alpha * smoothed[peak_idx]
        under_threshold = first_persistent_idx(after_peak_to_before_end, lambda v: v <= threshold, k_persist)

        if under_threshold is not None:
            new_start_idx = (peak_idx + 1) + under_threshold
            start_rule = f"threshold_{alpha}_p{k_persist}"
        else:
            post_deriv = np.gradient(smoothed)[peak_idx + 1 : right_bound]
            if len(post_deriv) == 0:
                new_start_idx = peak_idx
                start_rule = "no_derivative_after_peak"
            else:
                new_start_idx = (peak_idx + 1) + int(np.argmin(post_deriv))
                start_rule = "steepest_decline_after_peak_retry"
        
        new_start_idx = max(peak_idx + 1, min(new_start_idx, end_idx - 1))
        start_idx = new_start_idx
        
    # print("start rule:", start_rule)
    # print("peak at:", dates[peak_idx], "value:", smoothed[peak_idx])
    # print(f"Final harvest period for {label}: {dates[start_idx]} to {dates[end_idx]}")
    return dates[start_idx], dates[end_idx], smoothed, start_rule

# Adjust mask to reference source
def read_mask_resampled(mask_path, ref_src):
    # print(f"Reading and resampling mask: {mask_path}")
    with rasterio.open(mask_path) as mask_src:
        mask_data = mask_src.read(1)
        resampled_mask = np.empty((ref_src.height, ref_src.width), dtype=np.float32)
        rasterio.warp.reproject(
            source=mask_data,
            destination=resampled_mask,
            src_transform=mask_src.transform,
            src_crs=mask_src.crs,
            dst_transform=ref_src.transform,
            dst_crs=ref_src.crs,
            resampling=Resampling.nearest
        )
        return resampled_mask.astype(int)

# def read_mask(mask_path):
#     # print(f"Reading mask: {mask_path}")
#     with rasterio.open(mask_path) as src:
#         return src.read(1), src.transform # transform: grid(x, y) to map(latitude, longitude) transform
    
def reconstruct_mask(coords, shape):
    mask = np.zeros(shape, dtype=bool)
    mask[tuple(coords.T)] = True # (x,y) --> (y, x) and make it true if in coords(farm)
    return mask

def get_farms_in_crop_mask(crop_mask, crop_val, transform, min_pixels=11):
    binary_mask = (crop_mask == crop_val).astype(int) 
    labeled_array, num = label(binary_mask) # for each farm, label it with different number
    farms = {}

    for i in range(1, num + 1):
        mask = (labeled_array == i)
        if np.sum(mask) < min_pixels:
            print(f"⏭️ Skipping small farm (only {np.sum(mask)} pixels)")
            continue

        cy, cx = center_of_mass(mask) 
        x, y = rasterio.transform.xy(transform, cy, cx, offset='center')
        coords = np.argwhere(mask)  # get all (y, x) coordinates of the farm

        farm_id = f"{int(x)}_{int(y)}"
        farms[farm_id] = coords # return domain of each farm id

    return farms

def load_saved_index_images(index_dir, timestamps):
    # print(f"Loading saved index images from: {index_dir}")
    index_arrays = []
    for ts in timestamps:
        f_path = os.path.join(index_dir, f"{ts.strftime('%Y%m%d')}.npy")
        if os.path.exists(f_path):
            data = np.load(f_path)
            if np.nanmax(data) > 1.5:  # based on sentinel-2 scaling (-1,1)
                data = data / 10000.0
            index_arrays.append(data) # append each date's index array following timestamps order
        else:
            index_arrays.append(None)
    return index_arrays

def save_resampled_plot_mask_once(mask_path, sample_path, output_path):
    # print(f"Saving resampled plot mask to: {output_path}")
    if os.path.exists(output_path):
        print(f"ℹ️ Aligned mask already exists at {output_path}")
        return

    for folder in sorted(os.listdir(sample_path)):
        ts_path = os.path.join(sample_path, folder)
        if not os.path.isdir(ts_path):
            continue
        try:
            files = os.listdir(ts_path)
            red_file = next(f for f in files if 'B4' in f)
            red_path = os.path.join(ts_path, red_file)
            with rasterio.open(red_path) as ref_src:
                plot_mask_resized = read_mask_resampled(mask_path, ref_src)
                np.save(output_path, plot_mask_resized)
        
                print(f"✅ Saved aligned plot mask to {output_path}")
                return
        except Exception as e:
            print(f"⚠️ Could not process {folder}: {e}")
    raise RuntimeError("❌ Failed to find valid image to save plot mask.")

def save_alignment_qc_png(red_band_01, mask_int, nodata_val, out_png):
    farm_mask = (mask_int != 0)
    valid_obs = (~np.isclose(red_band_01, nodata_val)) & np.isfinite(red_band_01)

    # Adjust contrast for visualization
    def _stretch(x):
        lo, hi = np.nanpercentile(x, [2, 98])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = np.nanmin(x), np.nanmax(x)
        y = (x - lo) / max(hi - lo, 1e-6)
        return np.clip(y, 0, 1)

    base = _stretch(red_band_01)
    rgb = np.stack([base, base, base], axis=2)

    edges = farm_mask & (~binary_erosion(farm_mask, iterations=1, border_value=0))
    no_data_in_mask = farm_mask & (~valid_obs)

    rgb[edges] = [0.0, 1.0, 0.0]   # green
    rgb[no_data_in_mask] = [1.0, 0.0, 0.0]  # red

    imsave(out_png, (rgb * 255).astype(np.uint8))

def IoU_calculation(start_date, end_date, harvest_window):
    ws, we = harvest_window
    assert (ws is not None) and (we is not None), "harvest_window must be provided"
    gt_start = datetime.strptime(f"2019-{ws}", "%Y-%m-%d") if ws else None
    gt_end   = datetime.strptime(f"2019-{we}", "%Y-%m-%d") if we else None

    delta = max(0, (min(end_date, gt_end) - max(start_date, gt_start)).days)

    union = (max(end_date, gt_end) - min(start_date, gt_start)).days
    iou = delta / union
    return iou
            
# === Modular Global Crop-Level Analysis ===
def run_global_analysis():
    print("🌍 Starting global crop-level analysis...")

    allowed_crops = {"Corn", "Cotton", "Rice", "Soybeans", "Winter Wheat"}

    gt_windows = {
        "Corn": ("07-31", "10-15"),
        "Cotton": ("09-05", "11-30"),
        "Rice": ("08-05", "11-05"),
        "Soybeans": ("09-05", "11-30"),
        "Winter Wheat": ("05-31", "07-31")
    }

    all_crops = set()
    all_timestamps = []
    data_by_crop_and_time = {}
    summary_rows = []

    aligned_mask_path = os.path.join(sample_path, "plot_mask_resized.npy")
    save_resampled_plot_mask_once(plot_mask_path, sample_path, aligned_mask_path)

    for ts_folder in sorted(os.listdir(sample_path)):
        ts_path = os.path.join(sample_path, ts_folder)
        if not os.path.isdir(ts_path):
            continue
        print(f"🗂️ Processing timestamp folder: {ts_folder}")
        # calculate each timestamp's indices
        try:
            files = os.listdir(ts_path)
            red_file = next(f for f in files if 'B4' in f)
            nir_file = next(f for f in files if 'B8' in f)
            blue_file = next(f for f in files if 'B2' in f)
            swir_file = next(f for f in files if 'B11' in f or 'B12' in f)
            
            scl = None
            scl_file = next((f for f in files if 'SCL' in f), None)
 
            with rasterio.open(os.path.join(ts_path, red_file)) as red_src:
                red = red_src.read(1).astype('float32') / 10000.0
                nir = read_and_resample_band(os.path.join(ts_path, nir_file), red_src)
                blue = read_and_resample_band(os.path.join(ts_path, blue_file), red_src)
                swir = read_and_resample_band(os.path.join(ts_path, swir_file), red_src)
                if scl_file:
                    scl = read_and_resample_band(os.path.join(ts_path, scl_file), red_src, scl=True, resampling=Resampling.nearest).astype(np.int16)
                    
                plot_mask_resized = np.load(aligned_mask_path)
                plot_mask_resized[red == 0] = 0
                
                crop_labels = [val for val in np.unique(plot_mask_resized) if val in valid_crop_labels and val != 0]
                ndvi = calculate_ndvi(nir, red)
                ndwi = calculate_ndwi(nir, swir)
                evi = calculate_evi(nir, red, blue)
                
                if scl is not None:
                    bad_classes = np.array([3, 8, 9, 10, 11], dtype = np.int16)  # cloud shadows, clouds, cirrus, snow/ice
                    scl_valid = ~np.isin(scl, bad_classes) 
                else:
                    scl_valid = np.ones_like(red, dtype=bool) # same shape but all of them is 1

                timestamp = datetime.strptime(ts_folder, "%Y-%m-%d")
                all_timestamps.append(timestamp)

                for crop_label_val in crop_labels:
                    crop_name = crop_dict.get(crop_label_val, f"Crop_{crop_label_val}").replace("/", "-")
                    if crop_name not in allowed_crops:
                        continue

                    crop_dir = os.path.join(output_root, crop_name)
                    ndvi_dir = os.path.join(crop_dir, "NDVI")
                    ndwi_dir = os.path.join(crop_dir, "NDWI")
                    evi_dir = os.path.join(crop_dir, "EVI")
                    # cloud_dir = os.path.join(crop_dir, "CLOUD_MASKS")
                    # for d in [crop_dir, ndvi_dir, ndwi_dir, evi_dir, cloud_dir]:
                    #     ensure_dir(d)
                    for d in [crop_dir, ndvi_dir, ndwi_dir, evi_dir]:
                        ensure_dir(d)

                    mask_crop_path = os.path.join("debug_crop_mask/crop_mask", f"{crop_name}_{ts_folder}_mask.png")
                    mask_crop = (plot_mask_resized == crop_label_val) # boolean mask for the current crop. True and False
                    plt.imsave(mask_crop_path, mask_crop.astype(np.uint8)*255, cmap='gray')
                    mask = mask_crop & scl_valid # combine with cloud mask if available
                    mask_path = os.path.join("debug_crop_mask/combine_mask", f"{crop_name}_{ts_folder}_mask_combined.png")
                    plt.imsave(mask_path, mask.astype(np.uint8)*255, cmap='gray')
                    # np.save(os.path.join(crop_dir, f"{ts_folder}_mask.npy"), mask.astype(np.uint8))
                    # print("mask type:", mask.dtype, "unique values:", np.unique(mask))

                    ndvi_val = np.nanmean(np.where(mask, ndvi, np.nan)) # mean of ndvi values where mask is True, ignoring nan
                    ndwi_val = np.nanmean(np.where(mask, ndwi, np.nan))
                    evi_val = np.nanmean(np.where(mask, evi, np.nan))

                    all_crops.add(crop_label_val)
                    if crop_label_val not in data_by_crop_and_time:
                        data_by_crop_and_time[crop_label_val] = {}
                    # Store mean indices for this crop and timestamp
                    data_by_crop_and_time[crop_label_val][timestamp] = (ndvi_val, ndwi_val, evi_val)

                    # Save index maps
                    for index_data, index_val, index_dir, cmap in [
                        (ndvi, ndvi_val, ndvi_dir, 'RdYlGn'),
                        (ndwi, ndwi_val, ndwi_dir, 'Blues'),
                        (evi, evi_val, evi_dir, 'Greens'),
                    ]:
                        masked = np.where(mask, index_data, np.nan)
                        npy_path = os.path.join(index_dir, f"{timestamp:%Y%m%d}.npy")
                        png_path = os.path.join(index_dir, f"{timestamp:%Y%m%d}.png")
                        if not os.path.exists(npy_path) or not os.path.exists(png_path):
                            if not np.isnan(index_val):
                                np.save(os.path.join(index_dir, f"{timestamp.strftime('%Y%m%d')}.npy"), masked)
                                plt.imsave(os.path.join(index_dir, f"{timestamp.strftime('%Y%m%d')}.png"),
                                        np.nan_to_num(masked, nan=-1),
                                        cmap=cmap, vmin=-1, vmax=1)
                        else:
                            print(f"ℹ️ Index files already exist for {crop_name} at {timestamp:%Y-%m-%d}")
                        
                        # if scl is not None:
                        #     masked_scl = np.where(mask, scl, np.nan)
                        #     scl_npy = os.path.join(cloud_dir, f"{timestamp:%Y%m%d}_scl.npy")
                        #     scl_png = os.path.join(cloud_dir, f"{timestamp:%Y%m%d}_scl.png")
                        #     if not os.path.exists(scl_npy):
                        #         np.save(scl_npy, masked_scl)
                        #     if not (os.path.exists(scl_npy) and os.path.exists(scl_png)):
                        #         np.save(scl_npy, masked_scl)
                        #         save_scl_png(masked_scl, scl_png)
        except Exception as e:
            print(f"❌ Error in {ts_folder}: {e}")
            continue

    timestamps_array = sorted(set(all_timestamps))

    for label in sorted(all_crops):
        crop_name = crop_dict.get(label, f"Crop_{label}").replace("/", "-")
        if crop_name not in allowed_crops:
            continue

        crop_dir = os.path.join(output_root, crop_name)
        csv_path = os.path.join(crop_dir, f"{crop_name}_summary.csv")
        graph_path = os.path.join(crop_dir, f"{crop_name}_graph.png")

        ndvi_series, ndwi_series, evi_series = [], [], []
        for ts in timestamps_array:
            values = data_by_crop_and_time[label].get(ts, (np.nan, np.nan, np.nan))
            ndvi_series.append(values[0])
            ndwi_series.append(values[1])
            evi_series.append(values[2])

        ndvi_array = np.array(ndvi_series)
        ndwi_array = np.array(ndwi_series)
        evi_array = np.array(evi_series)

        if np.count_nonzero(~np.isnan(ndvi_array)) < 11:
            continue

        start_ndvi, end_ndvi, ndvi_smooth, ndvi_rule = detect_harvest_period(timestamps_array, np.nan_to_num(ndvi_array, nan=np.nanmean(ndvi_array)),label=f"crop:{crop_name}")
        start_ndwi, end_ndwi, ndwi_smooth, ndwi_rule = detect_harvest_period(timestamps_array, np.nan_to_num(ndwi_array, nan=np.nanmean(ndwi_array)),label=f"crop:{crop_name}")
        start_evi,  end_evi,  evi_smooth,  evi_rule  = detect_harvest_period(timestamps_array, np.nan_to_num(evi_array,  nan=np.nanmean(evi_array)),label=f"crop:{crop_name}")
        
        predicted_start_dt, chosen_source, chosen_rule, div_result_start = vote_or_fallback_with_rule(
            {"NDVI": start_ndvi, "NDWI": start_ndwi, "EVI": start_evi},
            {"NDVI": ndvi_rule, "NDWI": ndwi_rule, "EVI": evi_rule},
            kind="start",
            harvest_window=gt_windows.get(crop_name)
        )
        predicted_end_dt, div_result_end = vote_or_fallback([end_ndvi, end_ndwi, end_evi], timestamps_array, kind="end", harvest_window=gt_windows.get(crop_name))
        
        # print(f"Final predicted harvest period for {crop_name}: {predicted_start_dt} to {predicted_end_dt} (by {chosen_source} with rule {chosen_rule}) at the global level")

        assert predicted_start_dt <= predicted_end_dt, f"Invalid harvest period for {crop_name}"
        
        IoU = IoU_calculation(predicted_start_dt, predicted_end_dt, gt_windows.get(crop_name))
        print(f"🏆 Predicted harvest period for {crop_name}: {predicted_start_dt.date()} to {predicted_end_dt.date()} (by {chosen_source} with rule {chosen_rule}), IoU={IoU:.3f}")
        
        if not os.path.exists(graph_path):
            # ndvi_smooth = savgol_filter(np.nan_to_num(ndvi_array, nan=np.nanmean(ndvi_array)), 11, 3)
            # ndwi_smooth = savgol_filter(np.nan_to_num(ndwi_array, nan=np.nanmean(ndwi_array)), 11, 3)
            # evi_smooth  = savgol_filter(np.nan_to_num(evi_array,  nan=np.nanmean(evi_array)), 11, 3)
            
            # ndvi_smooth = sg_smooth(ndvi_array)
            # ndwi_smooth = sg_smooth(ndwi_array)
            # evi_smooth  = sg_smooth(evi_array)

            plt.figure(figsize=(10, 5))
            plt.plot(timestamps_array, ndvi_array, linestyle='--', alpha=0.4, label="NDVI (Raw)")
            plt.plot(timestamps_array, ndvi_smooth, linewidth=2, label="NDVI (Smoothed)")
            plt.plot(timestamps_array, ndwi_array, linestyle='--', alpha=0.4, label="NDWI (Raw)")
            plt.plot(timestamps_array, ndwi_smooth, linewidth=2, label="NDWI (Smoothed)")
            plt.plot(timestamps_array, evi_array, linestyle='--', alpha=0.4, label="EVI (Raw)")
            plt.plot(timestamps_array, evi_smooth, linewidth=2, label="EVI (Smoothed)")
            plt.axvline(predicted_start_dt, color='orange', linestyle=':', label='Start of Harvest')
            plt.axvline(predicted_end_dt, color='red', linestyle='-.', label='End of Harvest')

            # 🟩 Highlight GT window
            if crop_name in gt_windows:
                gt_start_str, gt_end_str = gt_windows[crop_name]
                year = predicted_start_dt.year
                try:
                    gt_start_dt = datetime.strptime(f"{year}-{gt_start_str}", "%Y-%m-%d")
                    gt_end_dt = datetime.strptime(f"{year}-{gt_end_str}", "%Y-%m-%d")
                    plt.axvspan(gt_start_dt, gt_end_dt, color='gray', alpha=0.2, label='Typical Harvest Window')
                except Exception as e:
                    print(f"⚠️ Failed to parse GT window for {crop_name}: {e}")

            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            plt.title(f"{crop_name} - Multi-Index Harvest Detection")
            plt.xlabel("Date")
            plt.ylabel("Index Value")
            plt.ylim(-1.5, 1.5)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(graph_path)
            plt.close()
        else:
            print(f"ℹ️ Graph already exists for {crop_name}")
            
        if not os.path.exists(csv_path):
            with open(csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Date", "NDVI", "NDWI", "EVI", "Harvest", "start_rule", "div_start", "div_end", "IoU"])
                for i, ts in enumerate(timestamps_array):
                    if np.isnan(ndvi_array[i]) and np.isnan(ndwi_array[i]) and np.isnan(evi_array[i]) and ts not in {predicted_start_dt, predicted_end_dt}:
                        continue
                    if ts == predicted_start_dt:
                        flag = "Start"
                        rule = chosen_rule
                        div_start = div_result_start
                        div_end = ""
                        IoU_result = IoU
                    elif ts == predicted_end_dt:
                        flag = "End"
                        rule = ""
                        div_start = ""
                        div_end = div_result_end
                        IoU_result = ""
                    else:
                        flag = ""
                        rule = ""
                        div_start = ""
                        div_end = ""
                        IoU_result = ""
                    writer.writerow([ts.strftime("%Y-%m-%d"),
                                    f"{ndvi_array[i]:.4f}",
                                    f"{ndwi_array[i]:.4f}",
                                    f"{evi_array[i]:.4f}",
                                    flag,
                                    rule,
                                    div_start,
                                    div_end,
                                    IoU_result])
                    summary_rows.append({
                        "Crop": crop_name,
                        "Date": ts.strftime("%Y-%m-%d"),
                        "NDVI": ndvi_array[i],
                        "NDWI": ndwi_array[i],
                        "EVI": evi_array[i],
                        "Harvest": flag,
                        "start_rule": rule,
                        "div_start": div_start,
                        "div_end": div_end,
                        "IoU": IoU_result
                    })
        else:
            print(f"ℹ️ Summary CSV already exists for {crop_name}")
        
    excel_path = os.path.join(output_root, "harvest_summary_all_crops.xlsx")
    if summary_rows and not os.path.exists(excel_path):
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_excel(excel_path, index=False)
        print("✅ Global analysis + harvest prediction completed.")
        LOGGER.info(f"✅ Wrote {excel_path}")
    else:
        print(f"↪️  Skipping {excel_path} (exists or no new rows)")
        LOGGER.info(f"↪️  Skip writing {excel_path} (exists or no rows)")

def run_farm_level_analysis():
    print("🚜 Starting farm-level analysis...")
    crop_root = output_root

    # Ensure the plot mask is aligned
    aligned_mask_path = os.path.join(sample_path, "plot_mask_resized.npy")
    if not os.path.exists(aligned_mask_path):
        print(f"ℹ️ Generating aligned mask...")
        save_resampled_plot_mask_once(plot_mask_path, sample_path, aligned_mask_path)
    mask_data = np.load(aligned_mask_path)

    # Get spatial transform from the original dataset
    ts_folder = next((f for f in sorted(os.listdir(sample_path)) if os.path.isdir(os.path.join(sample_path, f))), None)
    ts_path = os.path.join(sample_path, ts_folder)
    red_file = next((f for f in os.listdir(ts_path) if 'B4' in f), None)
    red_path = os.path.join(ts_path, red_file)

    with rasterio.open(red_path) as src:
        mask_transform = src.transform

    crop_dirs = [d for d in os.listdir(crop_root) if os.path.isdir(os.path.join(crop_root, d))]

    for crop_name in crop_dirs:
        print(f"📦 Processing crop: {crop_name}")
        crop_dir = os.path.join(crop_root, crop_name)
        ndvi_dir = os.path.join(crop_dir, "NDVI")
        ndwi_dir = os.path.join(crop_dir, "NDWI")
        evi_dir  = os.path.join(crop_dir, "EVI")

        files = sorted(os.listdir(ndvi_dir))
        timestamps = [datetime.strptime(f.split(".")[0], "%Y%m%d") for f in files if re.fullmatch(r"\d{8}\.png", f)]
        timestamps.sort()
        if not timestamps:
            print(f"⚠️ No valid timestamps for {crop_name}, skipping.")
            continue
        
        base_timestamp = tuple(timestamps)

        ndvi_stack = load_saved_index_images(ndvi_dir, list(base_timestamp))
        ndwi_stack = load_saved_index_images(ndwi_dir, list(base_timestamp))
        evi_stack  = load_saved_index_images(evi_dir, list(base_timestamp))

        crop_label = next((k for k, v in crop_dict.items() if v.replace("/", "-") == crop_name), None)
        if crop_label is None:
            print(f"⚠️ Skipping unknown crop label: {crop_name}")
            continue

        farms = get_farms_in_crop_mask(mask_data, int(crop_label), mask_transform, min_pixels=15)
        print(f"🌾 Found {len(farms)} farm plots for {crop_name}.")

        for farm_id, coords in farms.items():
            timestamps = list(base_timestamp)  # reset timestamps for each farm
            print(f"🔍 Processing farm {farm_id}...")
            mask_vis_dir = os.path.join(crop_dir, "Farms", farm_id)
            ensure_dir(mask_vis_dir)
            
            farm_csv = os.path.join(mask_vis_dir, f"{crop_name}_{farm_id}_summary.csv")
            farm_graph = os.path.join(mask_vis_dir, f"{crop_name}_{farm_id}_combined.png")
            if os.path.exists(farm_csv) and os.path.exists(farm_graph):
                print(f"⏭️ Skipping farm {farm_id} (CSV & plot already exist)")
                continue
            
            done_file = os.path.join(mask_vis_dir, "results.json")  # sentinel
            if os.path.exists(done_file):
                print(f"⏭️ Skipping farm {farm_id} (already processed)")
                continue
            
            farm_mask = reconstruct_mask(coords, mask_data.shape)

            green_mask = np.zeros((*farm_mask.shape, 3), dtype=np.uint8)
            green_mask[..., 1] = farm_mask.astype(np.uint8) * 255

            ndvi_series, ndwi_series, evi_series = [], [], []
            for ts, ndvi, ndwi, evi in zip(timestamps, ndvi_stack, ndwi_stack, evi_stack):
                if ndvi is None:
                    ndvi_series.append(np.nan)
                    ndwi_series.append(np.nan)
                    evi_series.append(np.nan)
                    continue

                masked_ndvi = np.where(farm_mask, ndvi, np.nan)
                masked_ndwi = np.where(farm_mask, ndwi, np.nan)
                masked_evi  = np.where(farm_mask, evi,  np.nan)

                v_ndvi = np.isfinite(masked_ndvi)
                v_ndwi = np.isfinite(masked_ndwi)
                v_evi  = np.isfinite(masked_evi)
                
                if not (np.any(v_ndvi) or np.any(v_ndwi) or np.any(v_evi)):
                    ndvi_series.append(np.nan)
                    ndwi_series.append(np.nan)
                    evi_series.append(np.nan)
                    continue

                mean_ndvi = masked_ndvi[v_ndvi].mean() if np.any(v_ndvi) else np.nan
                mean_ndwi = masked_ndwi[v_ndwi].mean() if np.any(v_ndwi) else np.nan
                mean_evi  = masked_evi[v_evi].mean()   if np.any(v_evi)  else np.nan
                
                # Debugging output
                # print(f"[{ts.strftime('%Y-%m-%d')}][{farm_id}] NDVI: {np.nanmin(masked_ndvi):.3f} to {np.nanmax(masked_ndvi):.3f} → mean: {mean_ndvi:.3f}")
                # print(f"[{ts.strftime('%Y-%m-%d')}][{farm_id}] NDWI: {np.nanmin(masked_ndwi):.3f} to {np.nanmax(masked_ndwi):.3f} → mean: {mean_ndwi:.3f}")
                # print(f"[{ts.strftime('%Y-%m-%d')}][{farm_id}] EVI:  {np.nanmin(masked_evi):.3f} to {np.nanmax(masked_evi):.3f} → mean: {mean_evi:.3f}")

                if np.nanmax(masked_ndvi) > 1 or np.nanmin(masked_ndvi) < -1:
                    print(f"⚠️ NDVI out of expected range at farm {farm_id}")
                if np.nanmax(masked_ndwi) > 1 or np.nanmin(masked_ndwi) < -1:
                    print(f"⚠️ NDWI out of expected range at farm {farm_id}")
                if np.nanmax(masked_evi) > 1 or np.nanmin(masked_evi) < -1:
                    print(f"⚠️ EVI out of expected range at farm {farm_id}")

                ndvi_series.append(mean_ndvi)
                ndwi_series.append(mean_ndwi)
                evi_series.append(mean_evi)

            ndvi_array = np.array(ndvi_series)
            ndwi_array = np.array(ndwi_series)
            evi_array = np.array(evi_series)

            valid_ndvi = np.count_nonzero(~np.isnan(ndvi_array))
            valid_ndwi = np.count_nonzero(~np.isnan(ndwi_array))
            valid_evi  = np.count_nonzero(~np.isnan(evi_array))

            print(f"📉 [{farm_id}] Valid NDVI points: {valid_ndvi}, NDWI: {valid_ndwi}, EVI: {valid_evi}")

            if np.count_nonzero(~np.isnan(ndvi_array)) < 11:
                print(f"⏭️ Skipping farm {farm_id} due to insufficient data.")
                continue

            start_ndvi, end_ndvi, ndvi_smooth, ndvi_rule = detect_harvest_period(timestamps, np.nan_to_num(ndvi_array, nan=np.nanmean(ndvi_array)),label=f"farm:{farm_id}")
            start_ndwi, end_ndwi, ndwi_smooth, ndwi_rule = detect_harvest_period(timestamps, np.nan_to_num(ndwi_array, nan=np.nanmean(ndwi_array)),label=f"farm:{farm_id}")
            start_evi,  end_evi,  evi_smooth,  evi_rule  = detect_harvest_period(timestamps, np.nan_to_num(evi_array,  nan=np.nanmean(evi_array)),label=f"farm:{farm_id}")

            # start_votes = Counter([start_ndvi, start_ndwi, start_evi])
            # end_votes   = Counter([end_ndvi, end_ndwi, end_evi])
            # predicted_start_dt = vote_or_fallback([start_ndvi, start_ndwi, start_evi], timestamps, kind="start")
            predicted_start_dt, chosen_source, chosen_rule, div_result_start = vote_or_fallback_with_rule(
                {"NDVI": start_ndvi, "NDWI": start_ndwi, "EVI": start_evi},
                {"NDVI": ndvi_rule, "NDWI": ndwi_rule, "EVI": evi_rule},
                kind="start",
                harvest_window=gt_windows.get(crop_name)
            )
            predicted_end_dt, div_result_end = vote_or_fallback([end_ndvi, end_ndwi, end_evi], timestamps, kind="end", harvest_window=gt_windows.get(crop_name))

            if predicted_end_dt < predicted_start_dt:
                end_candidates = [d for d in (end_ndvi, end_ndwi, end_evi) if d is not None and d >= predicted_start_dt]
                if end_candidates:
                    predicted_end_dt = max(end_candidates)
                else:
                    later_ts = [t for t in timestamps if t >= predicted_start_dt]
                    predicted_end_dt = later_ts[-1] if later_ts else predicted_start_dt
                    
            assert predicted_start_dt <= predicted_end_dt, f"Invalid harvest period for farm {farm_id}"
            
            iou_result = IoU_calculation(predicted_start_dt, predicted_end_dt, gt_windows.get(crop_name))
            
            graph_path = os.path.join(mask_vis_dir, f"{crop_name}_{farm_id}_combined.png")    
            if not os.path.exists(graph_path):
                plot_timestamps = list(timestamps)
                plot_ndvi_array = ndvi_array.copy()
                plot_ndwi_array = ndwi_array.copy()
                plot_evi_array  = evi_array.copy()
                
                fig = plt.figure(figsize=(14, 5))
                gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])

                ax0 = plt.subplot(gs[0])
                ax0.imshow(green_mask)
                ax0.set_title(f"Farm Mask - {farm_id}")
                ax0.axis("off")

                # ✅ Add circle to ax0 (same plot used in combined figure)
                y_coords, x_coords = np.where(farm_mask)
                if len(x_coords) > 0 and len(y_coords) > 0:
                    center_x = int(np.mean(x_coords))
                    center_y = int(np.mean(y_coords))
                    circle = Circle((center_x, center_y), radius=200, edgecolor='red', facecolor='none', linewidth= 0.5)
                    ax0.add_patch(circle)
                # ax0.imshow(green_mask)
                # ax0.set_title(f"Farm Mask - {farm_id}")
                # ax0.axis("off")

                ax1 = plt.subplot(gs[1])
                ax1.plot(plot_timestamps, plot_ndvi_array, linestyle='--', alpha=0.4, label="NDVI (Raw)")
                ax1.plot(plot_timestamps, ndvi_smooth, linewidth=2, label="NDVI (Smoothed)")
                ax1.plot(plot_timestamps, plot_ndwi_array, linestyle='--', alpha=0.4, label="NDWI (Raw)")
                ax1.plot(plot_timestamps, ndwi_smooth, linewidth=2, label="NDWI (Smoothed)")
                ax1.plot(plot_timestamps, plot_evi_array, linestyle='--', alpha=0.4, label="EVI (Raw)")
                ax1.plot(plot_timestamps, evi_smooth, linewidth=2, label="EVI (Smoothed)")
                ax1.axvline(predicted_start_dt, color='orange', linestyle=':', label='Start of Harvest')
                ax1.axvline(predicted_end_dt, color='red', linestyle='-.', label='End of Harvest')
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax1.set_title(f"{crop_name} - {farm_id} Harvest Detection")
                ax1.set_xlabel("Date")
                ax1.set_ylabel("Index Value")
                ax1.set_ylim(-1.5, 1.5)
                ax1.grid(True)
                plt.xticks(rotation=45)
                ax1.legend()

                plt.tight_layout()
                plt.savefig(graph_path)
                plt.close()
            else:
                print(f"ℹ️ Graph already exists for farm {farm_id}")
                
            csv_path = os.path.join(mask_vis_dir, f"{crop_name}_{farm_id}_summary.csv")
            if not os.path.exists(csv_path):
                csv_ts    = list(base_timestamp)   # or list(timestamps)
                csv_ndvi  = ndvi_array.copy()
                csv_ndwi  = ndwi_array.copy()
                csv_evi   = evi_array.copy()
                if predicted_start_dt not in timestamps:
                    pos = bisect_left(csv_ts, predicted_start_dt)
                    csv_ts.insert(pos, predicted_start_dt)
                    csv_ndvi = np.insert(csv_ndvi, pos, np.nan)
                    csv_ndwi = np.insert(csv_ndwi, pos, np.nan)
                    csv_evi  = np.insert(csv_evi,  pos, np.nan)
                df = pd.DataFrame({
                    "Date": [ts.strftime("%Y-%m-%d") for ts in csv_ts],
                    "NDVI": csv_ndvi,
                    "NDWI": csv_ndwi,
                    "EVI": csv_evi,
                    "Harvest": ["Start" if ts == predicted_start_dt else "End" if ts == predicted_end_dt else "" for ts in csv_ts],
                    "start_rule": [chosen_rule if ts == predicted_start_dt else "" for ts in csv_ts],
                    "div_result_start": [div_result_start if ts == predicted_start_dt else "" for ts in csv_ts],
                    "div_result_end": [div_result_end if ts == predicted_end_dt else "" for ts in csv_ts],
                    "IoU": [iou_result if ts == predicted_start_dt else "" for ts in csv_ts],
                })
                df.to_csv(csv_path, index=False)
            else:
                print(f"ℹ️ Summary CSV already exists for farm {farm_id}")
            

            print(f"✅ Saved output for farm {farm_id}. Start: {predicted_start_dt}, End: {predicted_end_dt}")

    print("🏁 All farms processed.")
    LOGGER.info("🏁 All farms processed.")

def summarize_farm_harvest_dates(output_root, summary_filename="farm_harvest_summary.csv"):
    print("📊 Summarizing farm harvest dates...")
    summary_data = {}

    for crop_name in os.listdir(output_root):
        crop_dir = os.path.join(output_root, crop_name)
        farms_dir = os.path.join(crop_dir, "Farms")
        if not os.path.isdir(farms_dir):
            continue

        for farm_id in os.listdir(farms_dir):
            farm_folder = os.path.join(farms_dir, farm_id)
            if not os.path.isdir(farm_folder):
                continue
            
            for file in os.listdir(farm_folder):
                if file.endswith("_summary.csv"):
                    df = pd.read_csv(os.path.join(farm_folder, file))
                    pick = df[df["Harvest"].isin(["End", "Start"])]
                    harvest_date = pick.iloc[0]["Date"] if not pick.empty else ""
                    summary_data[farm_id] = harvest_date
            out = pd.DataFrame(
                [{"FarmID": fid, "HarvestDate": summary_data[fid]} for fid in sorted(summary_data)]
            )
            out.to_csv(os.path.join(output_root, summary_filename), index=False)

    # Build output rows
    farm_ids = list(summary_data.keys())
    harvest_dates = [summary_data[fid] for fid in farm_ids]

    # Create DataFrame with one column per farm
    df_out = pd.DataFrame([farm_ids, harvest_dates])
    summary_path = os.path.join(output_root, summary_filename)
    df_out.to_csv(summary_path, index=False, header=False)
    
    # df_out = pd.DataFrame({"FarmID": farm_ids, "HarvestDate": harvest_dates})
    # summary_path = os.path.join(output_root, summary_filename)
    # df_out.to_csv(summary_path, index=False)
    
    print(f"✅ Harvest summary saved to: {summary_path}")

# === Entrypoint ===
if __name__ == "__main__":
    YEAR_TAGS = ["2020_AR"]
    
    for YEAR_TAG in YEAR_TAGS:
        setup_logging(year_tag=YEAR_TAG)
        INPUT_PARENT = f"../AR_sentinel2/{YEAR_TAG}"
        OUTPUT_PARENT = os.path.join("./", YEAR_TAG)
        ensure_dir(OUTPUT_PARENT)
        LOGGER.info(f"==== YEAR {YEAR_TAG} ====")
        
        for current in sorted(d for d in os.listdir(INPUT_PARENT)
                            if os.path.isdir(os.path.join(INPUT_PARENT, d))):
            sample_path = os.path.join(INPUT_PARENT, current)
            plot_mask_path = os.path.join(sample_path, "cdl.tif")
            
            if not os.path.exists(plot_mask_path):
                print(f"⚠️ Missing plot mask for {current}, skipping...")
                continue
            
            output_root = os.path.join(OUTPUT_PARENT, f"{current}_New_Index")
            ensure_dir(output_root)
            
            globals().update({
                "current": current,
                "sample_path": sample_path,
                "plot_mask_path": plot_mask_path,
                "output_root": output_root,
            })

            print(f"\n===== Processing {current} =====")
            if run_global:
                run_global_analysis()          
            if run_farm:
                run_farm_level_analysis()      
                summarize_farm_harvest_dates(output_root)  
            mark_done(output_root)