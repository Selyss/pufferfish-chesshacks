from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class CompactResidualBlock(nn.Module):
    """Light residual refinement with no normalization or dropout."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.lin1(x))
        y = self.lin2(y)
        return torch.relu(x + y)


class CompactNNUE(nn.Module):
    """Smaller NNUE variant without LayerNorms or Dropouts."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] | None = None,
        residual_repeats: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        hidden_dims = tuple(hidden_dims or (1024, 512, 256, 128))
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer.")

        if residual_repeats is None:
            residual_repeats = tuple(1 for _ in hidden_dims)
        elif len(residual_repeats) < len(hidden_dims):
            tail = residual_repeats[-1] if residual_repeats else 0
            residual_repeats = tuple(
                list(residual_repeats) + [tail] * (len(hidden_dims) - len(residual_repeats))
            )
        else:
            residual_repeats = tuple(residual_repeats[: len(hidden_dims)])

        self.hidden_dims = hidden_dims
        self.residual_repeats = residual_repeats

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim, repeat in zip(hidden_dims, residual_repeats):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            for _ in range(max(repeat, 0)):
                layers.append(CompactResidualBlock(hidden_dim))
            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.output_head = nn.Linear(prev_dim, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.backbone(inputs)
        return self.output_head(x).squeeze(-1)
