from __future__ import annotations

import argparse
import json
import random
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

from huggingface_hub import hf_hub_download
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .dataset import (
    DatasetConfig,
    FenFeatureEncoder,
    LightPreprocessedDataset,
    load_light_preprocessed_dataset,
)
from .model import SimpleNNUE
from .model_compact import CompactNNUE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an NNUE model on light-preprocessed data.")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--val-split", type=float, default=0.02)
    parser.add_argument("--limit-rows", type=int, default=None, help="Subset the dataset for quicker experiments.")
    parser.add_argument("--cache-dir", type=str, default=None, help="Custom Hugging Face cache directory.")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=None, help="Hidden layer sizes for the NNUE.")
    parser.add_argument("--residual-repeats", type=int, nargs="+", default=None, help="Residual block repeats per hidden layer.")
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", help="cuda, cpu, mps, or auto.")
    parser.add_argument("--output-dir", type=Path, default=Path("src/bot/checkpoints"))
    parser.add_argument("--label-key", type=str, choices=["cp_label", "eval_bucket"], default="cp_label")
    parser.add_argument("--target-scale", type=float, default=0.01, help="Scale applied to labels (1 centipawn = 0.01).")
    parser.add_argument("--skip-depth-feature", action="store_true")
    parser.add_argument("--skip-knodes-feature", action="store_true")
    parser.add_argument("--weight-by-depth", action="store_true")
    parser.add_argument("--weight-by-knodes", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Save checkpoints every N epochs.")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA devices.")
    parser.add_argument(
        "--model",
        type=str,
        choices=["simple", "compact"],
        default="simple",
        help="Choose between the original SimpleNNUE or the CompactNNUE.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=0,
        help="Log running train/val losses every N steps (0 disables).",
    )
    parser.add_argument(
        "--log-interval-eval",
        action="store_true",
        help="Also run validation when --log-interval triggers.",
    )
    return parser.parse_args()


def resolve_device(device_flag: str) -> torch.device:
    if device_flag != "auto":
        return torch.device(device_flag)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    use_amp: bool,
    grad_clip: float,
    epoch: int,
    log_interval: int = 0,
    val_loader: DataLoader | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    if use_amp:
        def autocast():
            return torch.autocast(device_type="cuda", dtype=torch.float16)
    else:
        def autocast():
            return nullcontext()

    for step, (features, labels, weights) in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        features = features.to(device, non_blocking=True)
        labels = labels.squeeze(-1).to(device, non_blocking=True)
        weights = weights.squeeze(-1).to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            preds = model(features)
            loss = ((preds - labels) ** 2) * weights
            loss = loss.mean()
        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        batch_size = features.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        if log_interval > 0 and step % log_interval == 0:
            avg_loss = total_loss / max(total_samples, 1)
            message = f"[epoch {epoch:02d} step {step:05d}] train_loss={avg_loss:.4f}"
            if val_loader is not None:
                val_loss = evaluate(model, val_loader, device)
                message += f" | val_loss={val_loss:.4f}"
                model.train()
            tqdm.write(message) 

    return total_loss / max(total_samples, 1)


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for features, labels, weights in tqdm(loader, desc="val", leave=False):
            features = features.to(device, non_blocking=True)
            labels = labels.squeeze(-1).to(device, non_blocking=True)
            weights = weights.squeeze(-1).to(device, non_blocking=True)
            preds = model(features)
            loss = ((preds - labels) ** 2) * weights
            batch_size = features.shape[0]
            total_loss += loss.mean().item() * batch_size
            total_samples += batch_size
    return total_loss / max(total_samples, 1)


def save_checkpoint(
    output_dir: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    stats: dict,
    model_config: dict,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"nnue_epoch{epoch:03d}.pt"
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "stats": stats,
        "model_config": model_config,
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def main() -> None:
    args = parse_args()
    if not 0.0 < args.val_split < 1.0:
        raise ValueError("val-split must be between 0 and 1.")
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    dataset_config = DatasetConfig(
        label_key=args.label_key,
        target_scale=args.target_scale,
        include_depth_feature=not args.skip_depth_feature,
        include_knodes_feature=not args.skip_knodes_feature,
        weight_by_depth=args.weight_by_depth,
        weight_by_knodes=args.weight_by_knodes,
    )

    file_path = hf_hub_download(
        repo_id="bantmen/lichess_all_mate_positions",
        filename="all_mate_positions.txt.gz",
        repo_type="dataset"
    )

    base_dataset = load_light_preprocessed_dataset(split="train", cache_dir=args.cache_dir)
    total_rows = len(base_dataset)
    if args.limit_rows:
        limit = min(args.limit_rows, total_rows)
        base_dataset = base_dataset.select(range(limit))
        print(f"Subsampled dataset to {limit} rows.")
    dataset = LightPreprocessedDataset(base_dataset, encoder=FenFeatureEncoder(), config=dataset_config, file_path=file_path)

    val_size = max(1, min(600000, int(len(dataset) * args.val_split)))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("Not enough samples for the requested validation split.")
    splits = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_set, val_set = splits

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=58,
        prefetch_factor=4,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        prefetch_factor=2,
        pin_memory=pin_memory,
    )

    if args.model == "simple":
        hidden_dims = tuple(args.hidden_dims) if args.hidden_dims else (2048, 2048, 1024, 512, 256)
        residual_repeats = (
            tuple(args.residual_repeats) if args.residual_repeats else tuple(2 for _ in hidden_dims)
        )
        model = SimpleNNUE(
            input_dim=dataset.feature_dim,
            hidden_dims=hidden_dims,
            dropout=args.dropout,
            residual_repeats=residual_repeats,
        ).to(device)
        model_hidden_dims = hidden_dims
        model_residual_repeats = residual_repeats
        model_dropout = args.dropout
    else:
        hidden_dims = tuple(args.hidden_dims) if args.hidden_dims else None
        residual_repeats = tuple(args.residual_repeats) if args.residual_repeats else None
        model = CompactNNUE(
            input_dim=dataset.feature_dim,
            hidden_dims=hidden_dims,
            residual_repeats=residual_repeats,
        ).to(device)
        model_hidden_dims = tuple(model.hidden_dims)
        model_residual_repeats = tuple(model.residual_repeats)
        model_dropout = 0.0

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

    use_amp = bool(args.amp and device.type == "cuda")
    try: 
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "training_config.json"
    model_config = {
        "model_type": args.model,
        "input_dim": dataset.feature_dim,
        "hidden_dims": list(model_hidden_dims) if model_hidden_dims is not None else None,
        "dropout": model_dropout,
        "residual_repeats": list(model_residual_repeats) if model_residual_repeats is not None else None,
    }
    config_payload = {
        "args": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "dataset": asdict(dataset_config),
        "model_config": model_config,
    }
    config_path.write_text(json.dumps(config_payload, indent=2))

    best_val = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler,
            use_amp,
            args.grad_clip,
            epoch,
            args.log_interval,
            val_loader if args.log_interval and args.log_interval_eval else None,
        )
        val_loss = evaluate(model, val_loader, device)
        scheduler.step(val_loss)
        duration = time.time() - start
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(
            f"Epoch {epoch:02d} | train={train_loss:.4f} | val={val_loss:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | {duration:.1f}s"
        )
        if val_loss < best_val:
            best_val = val_loss
            best_path = save_checkpoint(
                args.output_dir,
                epoch,
                model,
                optimizer,
                {"train_loss": train_loss, "val_loss": val_loss, "best": True},
                model_config,
            )
            print(f"  New best checkpoint saved to {best_path}")
        elif epoch % args.checkpoint_every == 0:
            ckpt_path = save_checkpoint(
                args.output_dir,
                epoch,
                model,
                optimizer,
                {"train_loss": train_loss, "val_loss": val_loss, "best": False},
                model_config,
            )
            print(f"  Checkpoint saved to {ckpt_path}")

    history_path = args.output_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2))
    print(f"Training complete. Best validation loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
