"""Stacked linear layer for phase-bucketed processing."""

from __future__ import annotations

import torch
import torch.nn as nn


class StackedLinear(nn.Module):
    """Stacked linear layer for phase-bucketed processing.

    This module maintains `count` independent linear layers (stacks) and applies
    a specific stack to each input sample based on the provided bucket index.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        count: Number of stacked layers (buckets).
    """

    def __init__(self, in_features: int, out_features: int, count: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.count = count
        self.linear = nn.Linear(in_features, out_features * count)

        self._init_uniformly()

    @torch.no_grad()
    def _init_uniformly(self) -> None:
        """Initialize all stacks with the same weights (copy from first stack)."""
        init_weight = self.linear.weight[: self.out_features, :].clone()
        init_bias = self.linear.bias[: self.out_features].clone()

        self.linear.weight.copy_(init_weight.repeat(self.count, 1))
        self.linear.bias.copy_(init_bias.repeat(self.count))

    def forward(self, x: torch.Tensor, ls_indices: torch.Tensor) -> torch.Tensor:
        """Forward pass with bucket selection."""
        stacked_output = self.linear(x)
        return self._select_output(stacked_output, ls_indices)

    def _select_output(
        self, stacked_output: torch.Tensor, ls_indices: torch.Tensor
    ) -> torch.Tensor:
        """Select output for each sample based on bucket index."""
        reshaped_output = stacked_output.reshape(-1, self.out_features)

        idx_offset = torch.arange(
            0,
            ls_indices.shape[0] * self.count,
            self.count,
            device=stacked_output.device,
        )
        indices = ls_indices + idx_offset

        return reshaped_output[indices]

    @torch.no_grad()
    def at_index(self, index: int) -> nn.Linear:
        """Extract a single stack as an independent nn.Linear module."""
        layer = nn.Linear(self.in_features, self.out_features)

        begin = index * self.out_features
        end = (index + 1) * self.out_features

        layer.weight.copy_(self.linear.weight[begin:end, :])
        layer.bias.copy_(self.linear.bias[begin:end])

        return layer
