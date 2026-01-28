"""Stacked linear layer for phase-bucketed processing."""

from __future__ import annotations

import torch
import torch.nn as nn


class StackedLinear(nn.Module):
    """Stacked linear layer for phase-bucketed processing.

    This module combines multiple linear layers with the same input/output
    dimensions into a single layer, selecting the appropriate output based
    on bucket indices. This pattern is inspired by Stockfish NNUE.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        count: Number of stacked layers (buckets).

    Example:
        >>> layer = StackedLinear(128, 64, count=6)
        >>> x = torch.randn(32, 128)  # batch of 32
        >>> indices = torch.randint(0, 6, (32,))  # bucket for each sample
        >>> output = layer(x, indices)  # (32, 64)
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
        """Forward pass with bucket selection.

        Args:
            x: Input tensor of shape (batch, in_features).
            ls_indices: Bucket indices of shape (batch,).

        Returns:
            Output tensor of shape (batch, out_features).
        """
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
        indices = ls_indices.flatten() + idx_offset

        return reshaped_output[indices]

    @torch.no_grad()
    def at_index(self, index: int) -> nn.Linear:
        """Extract a single stack as an independent nn.Linear module.

        Args:
            index: Stack index to extract.

        Returns:
            nn.Linear module with weights from the specified stack.
        """
        layer = nn.Linear(self.in_features, self.out_features)

        begin = index * self.out_features
        end = (index + 1) * self.out_features

        layer.weight.copy_(self.linear.weight[begin:end, :])
        layer.bias.copy_(self.linear.bias[begin:end])

        return layer
