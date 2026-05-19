# main.py
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import yaml

from config import Config
from log import PipelineLogger
from global_analysis import GlobalAnalyzer
from farm_analysis import FarmLevelAnalyzer
from summary import summarize_farm_harvest_dates
from utils.raster_io import cleanup_index_cache


def load_gt_windows_yaml(path: Path | None) -> dict[str, tuple[str, str]]:
    """
    Load GT windows mapping:

    Format A (recommended):
      Corn:
        - "09-10"
        - "10-05"

    Format B:
      Corn:
        start: "09-10"
        end: "10-05"
    """
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"GT windows YAML not found: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}

    gt: dict[str, tuple[str, str]] = {}
    for crop_name, v in data.items():
        if isinstance(v, (list, tuple)) and len(v) == 2:
            gt[crop_name] = (str(v[0]), str(v[1]))
        elif isinstance(v, dict) and "start" in v and "end" in v:
            gt[crop_name] = (str(v["start"]), str(v["end"]))
        else:
            raise ValueError(f"Bad GT window format for {crop_name}: {v}")
    return gt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Harvest estimation pipeline (refactored, multi-tile).")

    # Multi-tile root
    p.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root directory containing tile subdirectories (e.g. 12_0, 12_1, ...).",
    )

    # Config inputs
    p.add_argument(
        "--cdl-yaml",
        type=Path,
        required=True,
        help="CDL YAML mapping (num2class, crop_type).",
    )
    p.add_argument(
        "--gt-windows-yaml",
        type=Path,
        default=None,
        help="Optional YAML mapping crop_name -> [MM-DD, MM-DD] (or {start,end}).",
    )
    p.add_argument(
        "--cdl-root",
        type=Path,
        default=None,
        help=(
            "Optional root directory for CDL files. "
            "If set, per-tile CDL is resolved from <cdl-root>/<tile>/cdl.tif "
            "(or fallback <cdl-root>/<tile>.tif)."
        ),
    )

    # Outputs
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("./outputs"),
        help="Output root directory. Each tile gets its own subfolder here.",
    )

    # Year used for GT and voting
    p.add_argument("--year", type=int, default=2019)

    # Pipeline toggles
    p.add_argument("--no-global", action="store_true", help="Skip global analysis stage.")
    p.add_argument("--no-farm", action="store_true", help="Skip farm-level analysis stage.")
    p.add_argument(
        "--summarize",
        action="store_true",
        help="After farm stage, write farm_harvest_summary.csv per tile output folder.",
    )

    # Optional: run only selected tiles (otherwise process all)
    p.add_argument(
        "--tiles",
        nargs="*",
        default=None,
        help="Optional list of tile folder names to process (e.g. --tiles 12_0 12_1).",
    )

    # Logging
    p.add_argument("--log-dir", type=Path, default=Path("./logs"))
    p.add_argument("--log-name", type=str, default="harvest_pipeline")
    
    p.add_argument("--cleanup-npy", action="store_true")

    return p.parse_args()


def list_tile_dirs(dataset_root: Path, tiles: list[str] | None) -> list[Path]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"--dataset-root not found: {dataset_root}")
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"--dataset-root is not a directory: {dataset_root}")

    all_tiles = sorted([d for d in dataset_root.iterdir() if d.is_dir()])

    if tiles:
        tiles_set = set(tiles)
        selected = [d for d in all_tiles if d.name in tiles_set]
        missing = sorted(list(tiles_set - {d.name for d in selected}))
        if missing:
            raise FileNotFoundError(f"Requested tiles not found under {dataset_root}: {missing}")
        return selected

    return all_tiles


def main() -> None:
    args = parse_args()

    logger = PipelineLogger(name=args.log_name, log_dir=str(args.log_dir))

    gt_windows = load_gt_windows_yaml(args.gt_windows_yaml)

    cfg = Config(
        args.cdl_yaml,
        allowed_crops=set(gt_windows.keys()),
        output_root=str(args.output_root),
        gt_windows=gt_windows,
    )

    try:
        tile_dirs = list_tile_dirs(args.dataset_root, args.tiles)
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)

    logger.info("✅ Pipeline config loaded.")
    logger.info(f"dataset_root  = {args.dataset_root}")
    logger.info(f"tiles         = {len(tile_dirs)} ({'selected' if args.tiles else 'all'})")
    logger.info(f"output_root   = {args.output_root}")
    logger.info(f"cdl_root      = {args.cdl_root if args.cdl_root else '(use tile_dir/cdl.tif)'}")
    logger.info(f"year          = {args.year}")
    logger.info(f"run_global    = {not args.no_global}")
    logger.info(f"run_farm      = {not args.no_farm}")
    logger.info(f"summarize     = {args.summarize}")

    global_analyzer = GlobalAnalyzer(cfg, logger)
    farm_analyzer = FarmLevelAnalyzer(cfg, logger)

    processed = 0
    skipped = 0

    for tile_dir in tile_dirs:
        sample_path = tile_dir
        if args.cdl_root is None:
            plot_mask_path = tile_dir / "cdl.tif"
        else:
            cdl_in_tile_dir = args.cdl_root / tile_dir.name / "cdl.tif"
            cdl_flat_tif = args.cdl_root / f"{tile_dir.name}.tif"
            plot_mask_path = cdl_in_tile_dir if cdl_in_tile_dir.exists() else cdl_flat_tif
        tile_output_root = args.output_root / tile_dir.name
        tile_output_root.mkdir(parents=True, exist_ok=True)

        logger.info(f"🧩 Tile: {tile_dir.name}")
        logger.info(f"  sample_path   = {sample_path}")
        logger.info(f"  plot_mask_path= {plot_mask_path}")
        logger.info(f"  out_dir       = {tile_output_root}")

        if not plot_mask_path.exists():
            logger.warning(f"⚠️ Missing cdl.tif for tile {tile_dir.name}. Skipping.")
            skipped += 1
            continue

        # Global analysis
        if not args.no_global:
            logger.start(f"global:{tile_dir.name}")
            global_analyzer.run(
                sample_path=sample_path,
                plot_mask_path=plot_mask_path,
                output_root=tile_output_root,
                year=args.year,
                run_global=True,
            )
            logger.end(f"global:{tile_dir.name}")
        else:
            logger.info("↪️  Skipping global analysis (--no-global).")

        # Farm-level analysis
        if not args.no_farm:
            logger.start(f"farm:{tile_dir.name}")
            farm_analyzer.run(
                sample_path=sample_path,
                plot_mask_path=plot_mask_path,
                output_root=tile_output_root,
                year=args.year,
            )
            logger.end(f"farm:{tile_dir.name}")
        else:
            logger.info("↪️  Skipping farm-level analysis (--no-farm).")

        # Per-tile summary
        if args.summarize:
            logger.start(f"summary:{tile_dir.name}")
            summarize_farm_harvest_dates(str(tile_output_root))
            logger.end(f"summary:{tile_dir.name}")
        

        processed += 1
        
        if args.cleanup_npy:
            logger.info("🧹 Cleaning up index .npy cache files...")
            for crop_name in cfg.allowed_crops:
                crop_dir = tile_output_root / crop_name
                if crop_dir.exists():
                    cleanup_index_cache(crop_dir)
                    
    logger.info(f"🏁 Done. processed={processed}, skipped={skipped}")


if __name__ == "__main__":
    main()

# python main.py --dataset-root ../../AR_sentinel2/2019_AR --cdl-yaml ../configs/Arkansas/cdl.yaml --gt-windows-yaml /home/yikebe/DeepSatModels_updated/configs/Arkansas/gt_windows.yaml --output-root ../outputs/2019_AR_tmp --year 2019 --summarize --cleanup-npy
