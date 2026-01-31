"""Phase-adaptive sparse input layer for game phase-specific processing."""

from __future__ import annotations

from typing import Iterator

import torch
import torch.nn as nn

from models.model_common import SUM_OF_FEATURES, screlu
from sparse_linear import SparseLinear


class PhaseAdaptiveInput(nn.Module):
    """Phase-adaptive sparse input layer.

    Processes sparse features with different weights for different game phases.
    It maintains `count` independent sets of weights (buckets) and selects the
    appropriate one for each sample based on its game phase (ply).

    Args:
        count: Number of phase buckets.
        output_dim: Output dimension per bucket.
        max_ply: Maximum ply value for bucket size calculation.
        activation_scale: Scale factor for squared activation.
    """

    def __init__(
        self,
        count: int,
        output_dim: int,
        max_ply: int,
        activation_scale: float = 255 / 256,
    ) -> None:
        super().__init__()
        self.count = count
        self.output_dim = output_dim
        self.max_ply = max_ply
        self.activation_scale = activation_scale
        self.bucket_size = max_ply // count

        self.input = SparseLinear(SUM_OF_FEATURES, output_dim * count)
        self._init_layers()

    def _repeat_first_block(self, param: torch.Tensor, dim: int = 0) -> None:
        """Copy the first block along `dim` into all subsequent blocks."""
        if self.count <= 1:
            return

        first_block = param.narrow(dim, 0, self.output_dim).clone()
        for idx in range(1, self.count):
            param.narrow(dim, idx * self.output_dim, self.output_dim).copy_(first_block)

    def _bucket_lookup_indices(self, ply: torch.Tensor) -> torch.Tensor:
        """Return flattened indices that map each batch item to its phase bucket."""
        flat_ply = ply.view(-1)
        batch_size = flat_ply.size(0)
        ls_indices = flat_ply // self.bucket_size

        offsets = (
            torch.arange(batch_size, device=flat_ply.device, dtype=torch.long)
            * self.count
        )
        return ls_indices + offsets

    def _select_bucket(
        self, flat_source: torch.Tensor, bucket_indices: torch.Tensor
    ) -> torch.Tensor:
        """Slice the flattened tensor so each batch pulls the row matching its bucket."""
        return flat_source.reshape(-1, self.output_dim).index_select(0, bucket_indices)

    def _init_layers(self) -> None:
        """Initialize layers to ensure all phase buckets start identically."""
        with torch.no_grad():
            self._repeat_first_block(self.input.weight, dim=1)
            self._repeat_first_block(self.input.bias)

    def forward(self, feature_indices: torch.Tensor, ply: torch.Tensor) -> torch.Tensor:
        """Forward pass through phase-selected input layer."""
        x = self.input(feature_indices)
        bucket_indices = self._bucket_lookup_indices(ply)
        x = self._select_bucket(x, bucket_indices)
        return screlu(x, self.activation_scale)

    def get_layers(self) -> Iterator[nn.Linear]:
        """Extract individual phase-adaptive layers as separate nn.Linear modules."""
        for i in range(self.count):
            with torch.no_grad():
                layer = nn.Linear(SUM_OF_FEATURES, self.output_dim)
                start, end = i * self.output_dim, (i + 1) * self.output_dim
                layer.weight.data = self.input.weight[:, start:end].t()
                layer.bias.data = self.input.bias[start:end]
                yield layer
