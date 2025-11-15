from __future__ import annotations

import json
import struct
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bot.model import ResidualBlock, SimpleNNUE

LINEAR = 1
LAYERNORM = 2
RESIDUAL = 3


def load_model(checkpoint_path: str) -> SimpleNNUE:
    raw = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(raw, dict) and "model_state" in raw:
        state = raw["model_state"]
        config = raw.get("model_config") or {}
    else:
        state = raw
        config = {}
    input_dim = config.get("input_dim")
    hidden_dims = config.get("hidden_dims")
    dropout = config.get("dropout", 0.0)
    residual_repeats = config.get("residual_repeats")
    if input_dim is None:
        raise ValueError("Checkpoint missing model_config.input_dim")
    model = SimpleNNUE(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        residual_repeats=residual_repeats,
    )
    model.load_state_dict(state)
    model.eval()
    return model


def serialize_tensor(data: torch.Tensor) -> bytes:
    return data.detach().cpu().to(torch.float32).numpy().tobytes()


def collect_layers(model: SimpleNNUE) -> List[Tuple[int, dict]]:
    entries: List[Tuple[int, dict]] = []
    for module in model.backbone:
        if isinstance(module, nn.Linear):
            entries.append(
                (
                    LINEAR,
                    {
                        "in": module.in_features,
                        "out": module.out_features,
                        "weight": serialize_tensor(module.weight),
                        "bias": serialize_tensor(module.bias),
                    },
                )
            )
        elif isinstance(module, nn.LayerNorm):
            entries.append(
                (
                    LAYERNORM,
                    {
                        "size": module.normalized_shape[0],
                        "weight": serialize_tensor(module.weight),
                        "bias": serialize_tensor(module.bias),
                        "eps": module.eps,
                    },
                )
            )
        elif isinstance(module, ResidualBlock):
            entries.append(
                (
                    RESIDUAL,
                    {
                        "dim": module.lin1.out_features,
                        "weight1": serialize_tensor(module.lin1.weight),
                        "bias1": serialize_tensor(module.lin1.bias),
                        "weight2": serialize_tensor(module.lin2.weight),
                        "bias2": serialize_tensor(module.lin2.bias),
                        "norm_weight": serialize_tensor(module.norm.weight),
                        "norm_bias": serialize_tensor(module.norm.bias),
                        "eps": module.norm.eps,
                    },
                )
            )
    entries.append(
        (
            LINEAR,
            {
                "in": model.output_head.in_features,
                "out": model.output_head.out_features,
                "weight": serialize_tensor(model.output_head.weight),
                "bias": serialize_tensor(model.output_head.bias),
            },
        )
    )
    return entries


def export_bin(checkpoint_path: str, output_path: str) -> None:
    model = load_model(checkpoint_path)
    entries = collect_layers(model)
    metadata = {
        "format": "residual-nnue-v1",
        "input_dim": model.backbone[0].in_features,
        "layer_count": len(entries),
    }
    meta_bytes = json.dumps(metadata).encode("utf-8")
    with open(output_path, "wb") as f:
        f.write(struct.pack("<I", len(meta_bytes)))
        f.write(meta_bytes)
        f.write(struct.pack("<I", len(entries)))
        for type_id, payload in entries:
            f.write(struct.pack("<I", type_id))
            if type_id == LINEAR:
                f.write(struct.pack("<II", payload["out"], payload["in"]))
                f.write(payload["weight"])
                f.write(payload["bias"])
            elif type_id == LAYERNORM:
                f.write(struct.pack("<II", payload["size"], 0))
                f.write(payload["weight"])
                f.write(payload["bias"])
                f.write(struct.pack("<f", payload["eps"]))
            elif type_id == RESIDUAL:
                f.write(struct.pack("<II", payload["dim"], payload["dim"]))
                f.write(payload["weight1"])
                f.write(payload["bias1"])
                f.write(payload["weight2"])
                f.write(payload["bias2"])
                f.write(payload["norm_weight"])
                f.write(payload["norm_bias"])
                f.write(struct.pack("<f", payload["eps"]))
    size = Path(output_path).stat().st_size
    print(f"Exported {metadata['layer_count']} layers to {output_path} ({size/1024/1024:.2f} MB)")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m src.bot.export_int16 <checkpoint.pt> [output.bin]")
        raise SystemExit(1)
    ckpt = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) >= 3 else "nnue_residual.bin"
    export_bin(ckpt, out)


if __name__ == "__main__":
    main()
