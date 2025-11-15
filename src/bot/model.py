from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = torch.relu(self.lin1(x))
        y = self.dropout(y)
        y = self.lin2(y)
        out = residual + y
        return torch.relu(self.norm(out))


class SimpleNNUE(nn.Module):
    """Configurable NNUE with optional residual refinement blocks."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] | None = None,
        dropout: float = 0.0,
        residual_repeats: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        hidden_dims = tuple(hidden_dims or (2048, 2048, 1024, 512, 256))
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer.")
        if residual_repeats is None:
            residual_repeats = tuple(2 for _ in hidden_dims)
        elif len(residual_repeats) < len(hidden_dims):
            tail = residual_repeats[-1] if residual_repeats else 1
            residual_repeats = tuple(
                list(residual_repeats) + [tail] * (len(hidden_dims) - len(residual_repeats))
            )
        else:
            residual_repeats = tuple(residual_repeats[: len(hidden_dims)])

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim, repeat in zip(hidden_dims, residual_repeats):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            for _ in range(max(repeat, 0)):
                layers.append(ResidualBlock(hidden_dim, dropout=dropout))
            prev_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.output_head = nn.Linear(prev_dim, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.backbone(inputs)
        return self.output_head(x).squeeze(-1)
