from __future__ import annotations

import argparse
from pathlib import Path

from doy_prediction.tile_cnn_data import (
    build_tile_crop_records,
    load_crop_windows_yaml,
)
from doy_prediction.tile_rnn_model import TileRNNRegressor
from doy_prediction.train_tile_cnn import (
    DEFAULT_CROPS,
    resolve_device,
    set_seed,
    train_crop_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train crop-specific RNN harvest window regressors.")
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--crops", nargs="+", default=list(DEFAULT_CROPS))
    parser.add_argument("--feature-set", choices=["ndvi_only", "all_indices"], default="all_indices")
    parser.add_argument("--train-years", nargs="+", type=int, required=True)
    parser.add_argument("--test-years", nargs="+", type=int, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--min-points", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument(
        "--hidden-sizes",
        nargs="+",
        type=int,
        default=None,
        help="Optional per-layer hidden sizes, for example: --hidden-sizes 32 64 128.",
    )
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--rnn-type", choices=["gru", "lstm", "rnn"], default="gru")
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument(
        "--log-test-every-epoch",
        action="store_true",
        help="Log test metrics to wandb every epoch for charting. Does not affect model selection.",
    )
    parser.add_argument(
        "--crop-windows-yaml",
        type=Path,
        default=None,
        help="Optional YAML mapping crop names to fixed input windows [MM-DD, MM-DD].",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", type=str, default="DeepSatModels-harvest")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="online",
        help="Wandb logging mode when --wandb is enabled.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    crop_windows = load_crop_windows_yaml(args.crop_windows_yaml)

    all_years = sorted(set(args.train_years) | set(args.test_years))
    records = build_tile_crop_records(
        args.outputs_root,
        crops=args.crops,
        years=all_years,
        feature_set=args.feature_set,
        min_points=args.min_points,
        require_labels=True,
        crop_windows=crop_windows,
    )
    if not records:
        raise RuntimeError("No tile-crop records found for the requested configuration.")

    model_config = {
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "hidden_sizes": args.hidden_sizes,
        "dropout": args.dropout,
        "rnn_type": args.rnn_type,
        "bidirectional": args.bidirectional,
    }

    for crop in args.crops:
        crop_records = [record for record in records if record.crop == crop]
        if not crop_records:
            print(f"[skip] No samples found for crop={crop}")
            continue

        train_crop_model(
            crop_records,
            crop=crop,
            feature_set=args.feature_set,
            save_dir=args.save_dir,
            train_years=args.train_years,
            test_years=args.test_years,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            val_fraction=args.val_fraction,
            num_workers=args.num_workers,
            seed=args.seed,
            device=device,
            wandb_enabled=args.wandb,
            log_test_every_epoch=args.log_test_every_epoch,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_run_name=args.wandb_run_name,
            wandb_tags=args.wandb_tags,
            wandb_mode=args.wandb_mode,
            outputs_root=args.outputs_root,
            min_points=args.min_points,
            model_factory=lambda in_channels: TileRNNRegressor(
                in_channels=in_channels,
                **model_config,
            ),
            model_name="rnn",
            model_config=model_config,
        )


if __name__ == "__main__":
    main()
