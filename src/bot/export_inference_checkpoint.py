from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader

try:
    from .dataset import (
        DatasetConfig,
        FenFeatureEncoder,
        LightPreprocessedDataset,
        load_light_preprocessed_dataset,
    )
    from .model import SimpleNNUE
    from .model_compact import CompactNNUE
except ImportError:  # Allow running as `python src/bot/export_inference_checkpoint.py`
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))
    from bot.dataset import (
        DatasetConfig,
        FenFeatureEncoder,
        LightPreprocessedDataset,
        load_light_preprocessed_dataset,
    )
    from bot.model import SimpleNNUE
    from bot.model_compact import CompactNNUE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strip a training checkpoint down to inference-only weights.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("src/bot/checkpoints/modal-compact-a100/nnue_epoch001.pt"),
        help="Path to the training checkpoint produced by train_nnue.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination for the inference checkpoint (defaults to <input>_inference.pt).",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Quantize floating-point tensors to float16 before saving.",
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=None,
        help="Optional path to training_config.json for dataset metadata (defaults to alongside the checkpoint).",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=0,
        help="When --fp16 is set, evaluate both precisions on this many samples (0 to skip).",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=256,
        help="Batch size for the optional evaluation pass.",
    )
    parser.add_argument(
        "--eval-cache-dir",
        type=str,
        default=None,
        help="Custom cache directory for Hugging Face datasets during evaluation.",
    )
    return parser.parse_args()


def load_training_checkpoint(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Unexpected checkpoint format: expected dict payload.")
    model_state = payload.get("model_state")
    if model_state is None:
        raise ValueError("Checkpoint missing 'model_state'.")
    model_config = payload.get("model_config")
    if model_config is None:
        raise ValueError("Checkpoint missing 'model_config'.")
    return {"model_state": model_state, "model_config": model_config}


def quantize_state_dict_fp16(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    quantized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value) and value.is_floating_point():
            quantized[key] = value.detach().to(torch.float16)
        else:
            quantized[key] = value
    return quantized


def read_training_config(path: Optional[Path]) -> dict[str, Any]:
    if path and path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"Warning: failed to parse training config at {path}, using defaults.")
    return {}


def ensure_dataset_config(config_payload: dict[str, Any]) -> DatasetConfig:
    dataset_cfg = config_payload.get("dataset") if config_payload else None
    if isinstance(dataset_cfg, dict):
        return DatasetConfig(**dataset_cfg)
    return DatasetConfig()


def build_model_from_config(model_config: dict[str, Any]) -> torch.nn.Module:
    model_type = model_config.get("model_type", "simple")
    input_dim = model_config.get("input_dim")
    hidden_dims = model_config.get("hidden_dims")
    residual_repeats = model_config.get("residual_repeats")
    if model_type == "compact":
        return CompactNNUE(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            residual_repeats=residual_repeats,
        )
    dropout = model_config.get("dropout", 0.0)
    return SimpleNNUE(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        residual_repeats=residual_repeats,
    )


def build_eval_loader(
    dataset_config: DatasetConfig,
    cache_dir: Optional[str],
    sample_count: int,
    batch_size: int,
) -> DataLoader:
    base_dataset = load_light_preprocessed_dataset(split="train", cache_dir=cache_dir)
    limit = min(sample_count, len(base_dataset))
    base_dataset = base_dataset.select(range(limit))
    dataset = LightPreprocessedDataset(base_dataset, encoder=FenFeatureEncoder(), config=dataset_config)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def evaluate_model_state(
    model_config: dict[str, Any],
    state_dict: dict[str, torch.Tensor],
    loader: DataLoader,
    precision: torch.dtype,
) -> tuple[float, torch.Tensor]:
    device = torch.device("cpu")
    model = build_model_from_config(model_config).to(device)
    if precision == torch.float16:
        model = model.half()
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    total_loss = 0.0
    total_weight = 0.0
    preds_all: list[torch.Tensor] = []
    with torch.no_grad():
        for features, labels, weights in loader:
            features = features.to(device)
            if precision == torch.float16:
                features = features.half()
            labels = labels.squeeze(-1).to(device)
            weights = weights.squeeze(-1).to(device)
            outputs = model(features)
            preds = outputs
            preds_all.append(preds.detach().cpu().float())
            loss = ((preds - labels) ** 2) * weights
            total_loss += loss.sum().item()
            total_weight += weights.sum().item()
    stacked_preds = torch.cat(preds_all, dim=0).squeeze(-1)
    avg_loss = total_loss / max(total_weight, 1.0)
    return avg_loss, stacked_preds


def evaluate_precision_change(
    model_config: dict[str, Any],
    baseline_state: dict[str, torch.Tensor],
    quantized_state: dict[str, torch.Tensor],
    dataset_config: DatasetConfig,
    cache_dir: Optional[str],
    sample_count: int,
    batch_size: int,
) -> dict[str, float]:
    loader = build_eval_loader(dataset_config, cache_dir, sample_count, batch_size)
    fp32_loss, fp32_preds = evaluate_model_state(model_config, baseline_state, loader, torch.float32)
    fp16_loss, fp16_preds = evaluate_model_state(model_config, quantized_state, loader, torch.float16)
    diff = (fp16_preds - fp32_preds).abs()
    return {
        "fp32_loss": fp32_loss,
        "fp16_loss": fp16_loss,
        "loss_delta": fp16_loss - fp32_loss,
        "avg_abs_pred_delta": diff.mean().item(),
        "max_abs_pred_delta": diff.max().item(),
    }


def main() -> None:
    args = parse_args()
    suffix = "_fp16_inference" if args.fp16 else "_inference"
    output_path = args.output or args.input.with_name(args.input.stem + f"{suffix}.pt")
    inference_payload = load_training_checkpoint(args.input)
    training_config_path = args.training_config or (args.input.parent / "training_config.json")
    training_config_payload = read_training_config(training_config_path)
    baseline_state = {
        key: value.detach().clone() if torch.is_tensor(value) else value
        for key, value in inference_payload["model_state"].items()
    }
    if args.fp16:
        quantized_state = quantize_state_dict_fp16(inference_payload["model_state"])
        inference_payload["model_state"] = quantized_state
    else:
        quantized_state = inference_payload["model_state"]
    torch.save(inference_payload, output_path)
    config = inference_payload["model_config"]
    print("✓ Wrote inference checkpoint")
    print(f"  Input:  {args.input}")
    print(f"  Output: {output_path}")
    dtype_label = "fp16" if args.fp16 else "fp32"
    print(
        f"  Model:  {config.get('model_type', 'simple')} | precision={dtype_label} | "
        f"input_dim={config.get('input_dim')} | hidden_dims={config.get('hidden_dims')}"
    )
    if args.fp16 and args.eval_samples > 0:
        dataset_config = ensure_dataset_config(training_config_payload)
        eval_summary = evaluate_precision_change(
            model_config=config,
            baseline_state=baseline_state,
            quantized_state=quantized_state,
            dataset_config=dataset_config,
            cache_dir=args.eval_cache_dir,
            sample_count=args.eval_samples,
            batch_size=args.eval_batch_size,
        )
        print("  Evaluation (weighted MSE on sample):")
        print(f"    fp32_loss = {eval_summary['fp32_loss']:.6f}")
        print(f"    fp16_loss = {eval_summary['fp16_loss']:.6f} (Δ {eval_summary['loss_delta']:+.6e})")
        print(
            f"    avg|Δpred| = {eval_summary['avg_abs_pred_delta']:.6e}, "
            f"max|Δpred| = {eval_summary['max_abs_pred_delta']:.6e}"
        )
        print(f"    Samples evaluated: {args.eval_samples}")


if __name__ == "__main__":
    main()
