from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

from doy_prediction.tile_cnn_data import (
    TileCropDataset,
    TileCropRecord,
    build_tile_crop_records,
    get_feature_names,
    load_crop_windows_yaml,
)
from doy_prediction.metrics import maybe_harvest_window_iou
from doy_prediction.tile_cnn_model import TileCNNRegressor, normalized_to_doy


DEFAULT_CROPS = ("Corn", "Rice", "Soybeans")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train crop-specific 1D-CNN harvest window regressors.")
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
        )


def train_crop_model(
    records: list[TileCropRecord],
    *,
    crop: str,
    feature_set: str,
    save_dir: Path,
    train_years: Iterable[int],
    test_years: Iterable[int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    val_fraction: float,
    num_workers: int,
    seed: int,
    device: torch.device,
    wandb_enabled: bool,
    log_test_every_epoch: bool,
    wandb_project: str,
    wandb_entity: str | None,
    wandb_run_name: str | None,
    wandb_tags: list[str] | None,
    wandb_mode: str,
    outputs_root: Path,
    min_points: int,
) -> None:
    train_years = set(int(year) for year in train_years)
    test_years = set(int(year) for year in test_years)

    train_records = [record for record in records if record.year in train_years]
    test_records = [record for record in records if record.year in test_years]
    if not train_records:
        print(f"[skip] No training samples found for crop={crop}")
        return
    if not test_records:
        print(f"[warn] No test samples found for crop={crop}")

    train_records, val_records = split_train_val(train_records, val_fraction=val_fraction, seed=seed)

    run_dir = save_dir / feature_set / crop.replace("/", "_")
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader = make_loader(train_records, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = make_loader(val_records, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = make_loader(test_records, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    in_channels = len(get_feature_names(feature_set)) + 1
    model = TileCNNRegressor(in_channels=in_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.L1Loss()

    best_state = None
    best_score = float("inf")
    history = []
    wandb_run = init_wandb_run(
        enabled=wandb_enabled,
        project=wandb_project,
        entity=wandb_entity,
        run_name=wandb_run_name,
        tags=wandb_tags,
        mode=wandb_mode,
        crop=crop,
        feature_set=feature_set,
        train_years=sorted(train_years),
        test_years=sorted(test_years),
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        val_fraction=val_fraction,
        seed=seed,
        num_workers=num_workers,
        device=str(device),
        outputs_root=str(outputs_root),
        save_dir=str(save_dir),
        min_points=min_points,
        num_train=len(train_records),
        num_val=len(val_records),
        num_test=len(test_records),
        in_channels=in_channels,
        log_test_every_epoch=log_test_every_epoch,
    )

    try:
        for epoch in range(1, epochs + 1):
            train_loss = run_epoch(model, train_loader, optimizer, loss_fn, device)
            val_metrics = evaluate_model(model, val_loader, device) if val_records else {}
            test_epoch_metrics = (
                evaluate_model(model, test_loader, device)
                if log_test_every_epoch and test_records
                else {}
            )
            val_score = val_metrics.get("mae_mean", train_loss)
            epoch_row = {
                "epoch": epoch,
                "train_loss": train_loss,
                **{f"val_{k}": v for k, v in val_metrics.items()},
                **{f"test_{k}": v for k, v in test_epoch_metrics.items()},
            }
            history.append(epoch_row)
            if val_score <= best_score:
                best_score = val_score
                best_state = {
                    "model_state": model.state_dict(),
                    "crop": crop,
                    "feature_set": feature_set,
                    "in_channels": in_channels,
                    "epochs": epoch,
                }
            log_wandb_epoch(
                wandb_run,
                epoch=epoch,
                crop=crop,
                feature_set=feature_set,
                train_loss=train_loss,
                val_metrics=val_metrics,
                test_metrics=test_epoch_metrics,
                best_val_mae=best_score,
            )
            print(
                f"[{crop}][{feature_set}] epoch={epoch}/{epochs} train_loss={train_loss:.4f}"
                + (f" val_mae={val_metrics['mae_mean']:.2f}" if val_metrics else "")
                + (
                    f" test_mae={test_epoch_metrics['mae_mean']:.2f}"
                    if test_epoch_metrics
                    else ""
                )
            )

        if best_state is None:
            best_state = {
                "model_state": model.state_dict(),
                "crop": crop,
                "feature_set": feature_set,
                "in_channels": in_channels,
                "epochs": epochs,
            }

        checkpoint_path = run_dir / "best_model.pt"
        torch.save(best_state, checkpoint_path)

        model.load_state_dict(best_state["model_state"])
        train_metrics = evaluate_model(
            model,
            make_loader(train_records, batch_size=batch_size, shuffle=False, num_workers=num_workers),
            device,
        )
        val_metrics = evaluate_model(model, val_loader, device) if val_records else {}
        test_metrics = evaluate_model(model, test_loader, device) if test_records else {}

        write_predictions_csv(run_dir / "train_predictions.csv", predict_records(model, train_loader, device), split="train")
        if val_records:
            write_predictions_csv(run_dir / "val_predictions.csv", predict_records(model, val_loader, device), split="val")
        if test_records:
            write_predictions_csv(run_dir / "test_predictions.csv", predict_records(model, test_loader, device), split="test")

        metrics = {
            "crop": crop,
            "feature_set": feature_set,
            "train_years": sorted(train_years),
            "test_years": sorted(test_years),
            "num_train": len(train_records),
            "num_val": len(val_records),
            "num_test": len(test_records),
            "history": history,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "checkpoint": str(checkpoint_path),
        }
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        finalize_wandb_run(
            wandb_run,
            checkpoint_path=checkpoint_path,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            best_epoch=int(best_state["epochs"]),
            run_dir=run_dir,
        )
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def split_train_val(
    records: list[TileCropRecord],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[TileCropRecord], list[TileCropRecord]]:
    if len(records) < 2 or val_fraction <= 0:
        return records, []

    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    val_size = max(1, int(round(len(shuffled) * val_fraction)))
    val_size = min(val_size, len(shuffled) - 1)
    return shuffled[val_size:], shuffled[:val_size]


def make_loader(
    records: list[TileCropRecord],
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        TileCropDataset(records),
        batch_size=batch_size,
        shuffle=shuffle if records else False,
        num_workers=num_workers,
    )


def run_epoch(
    model: TileCNNRegressor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    if len(loader.dataset) == 0:
        return 0.0

    model.train()
    loss_sum = 0.0
    count = 0
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        batch_size = x.shape[0]
        loss_sum += loss.item() * batch_size
        count += batch_size
    return loss_sum / max(count, 1)


@torch.no_grad()
def predict_records(
    model: TileCNNRegressor,
    loader: DataLoader,
    device: torch.device,
) -> list[dict[str, object]]:
    predictions: list[dict[str, object]] = []
    model.eval()
    for batch in loader:
        x = batch["x"].to(device)
        pred = model(x)
        pred_doy = normalized_to_doy(pred)

        batch_size = x.shape[0]
        for idx in range(batch_size):
            true_start_doy = tensor_item_to_optional_int(batch["start_doy"][idx])
            true_end_doy = tensor_item_to_optional_int(batch["end_doy"][idx])
            pred_start_doy = int(pred_doy[idx, 0])
            pred_end_doy = int(pred_doy[idx, 1])
            predictions.append(
                {
                    "year": int(batch["year"][idx]),
                    "tile": str(batch["tile"][idx]),
                    "crop": str(batch["crop"][idx]),
                    "true_start_doy": none_to_blank(true_start_doy),
                    "true_end_doy": none_to_blank(true_end_doy),
                    "pred_start_doy": pred_start_doy,
                    "pred_end_doy": pred_end_doy,
                    "window_iou": none_to_blank(
                        maybe_harvest_window_iou(
                            pred_start_doy=pred_start_doy,
                            pred_end_doy=pred_end_doy,
                            true_start_doy=true_start_doy,
                            true_end_doy=true_end_doy,
                        )
                    ),
                    "feature_set": str(batch["feature_set"][idx]),
                    "source_workbook": str(batch["source_workbook"][idx]),
                }
            )
    return predictions


@torch.no_grad()
def evaluate_model(
    model: TileCNNRegressor,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    predictions = predict_records(model, loader, device)
    if not predictions:
        return {}

    true_start = np.array([int(row["true_start_doy"]) for row in predictions], dtype=np.float32)
    true_end = np.array([int(row["true_end_doy"]) for row in predictions], dtype=np.float32)
    pred_start = np.array([int(row["pred_start_doy"]) for row in predictions], dtype=np.float32)
    pred_end = np.array([int(row["pred_end_doy"]) for row in predictions], dtype=np.float32)
    window_iou = np.array([float(row["window_iou"]) for row in predictions], dtype=np.float32)

    start_mae = float(np.mean(np.abs(pred_start - true_start)))
    end_mae = float(np.mean(np.abs(pred_end - true_end)))
    window_mae = float(np.mean(np.abs((pred_end - pred_start) - (true_end - true_start))))
    ordered_pct = float(np.mean(pred_start <= pred_end))
    iou_mean = float(np.mean(window_iou))
    return {
        "mae_start": start_mae,
        "mae_end": end_mae,
        "mae_mean": (start_mae + end_mae) / 2.0,
        "mae_window": window_mae,
        "iou_mean": iou_mean,
        "ordered_pct": ordered_pct,
    }


def write_predictions_csv(path: Path, rows: list[dict[str, object]], *, split: str) -> None:
    import csv

    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["split"] + list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({"split": split, **row})


def resolve_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def none_to_blank(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, torch.Tensor):
        return value.item()
    return value


def tensor_item_to_optional_int(value: object) -> int | None:
    value = none_to_blank(value)
    if value == "":
        return None
    return int(value)


def init_wandb_run(
    *,
    enabled: bool,
    project: str,
    entity: str | None,
    run_name: str | None,
    tags: list[str] | None,
    mode: str,
    **config: object,
):
    if not enabled:
        return None
    if wandb is None:
        raise RuntimeError("wandb is not installed, but --wandb was requested.")

    default_run_name = f"{config['crop']}-{config['feature_set']}"
    return wandb.init(
        project=project,
        entity=entity,
        name=run_name or default_run_name,
        tags=list(tags) if tags is not None else None,
        mode=mode,
        config=config,
    )


def log_wandb_epoch(
    wandb_run,
    *,
    epoch: int,
    crop: str,
    feature_set: str,
    train_loss: float,
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    best_val_mae: float,
) -> None:
    if wandb_run is None:
        return

    payload = {
        "epoch": epoch,
        "train/loss": train_loss,
        "best/val_mae_mean": best_val_mae,
        "meta/crop": crop,
        "meta/feature_set": feature_set,
    }
    for key, value in val_metrics.items():
        payload[f"val/{key}"] = value
    for key, value in test_metrics.items():
        payload[f"test/{key}"] = value
    wandb_run.log(payload, step=epoch)


def finalize_wandb_run(
    wandb_run,
    *,
    checkpoint_path: Path,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    best_epoch: int,
    run_dir: Path,
) -> None:
    if wandb_run is None:
        return

    final_payload: dict[str, object] = {
        "final/best_epoch": best_epoch,
    }
    for prefix, metrics in (("train", train_metrics), ("val", val_metrics), ("test", test_metrics)):
        for key, value in metrics.items():
            final_payload[f"{prefix}/{key}"] = value
    if len(final_payload) > 1:
        wandb_run.log(final_payload)

    wandb_run.summary["best_epoch"] = best_epoch
    wandb_run.summary["checkpoint_path"] = str(checkpoint_path)
    wandb_run.summary["run_dir"] = str(run_dir)
    for prefix, metrics in (("train", train_metrics), ("val", val_metrics), ("test", test_metrics)):
        for key, value in metrics.items():
            wandb_run.summary[f"{prefix}/{key}"] = value


if __name__ == "__main__":
    main()
