"""WASM-compatible Reversi neural network model."""

from __future__ import annotations

from typing import Iterator

import torch
import torch.nn as nn

from models.model_common import (
    FEATURE_CUM_OFFSETS,
    SUM_OF_FEATURES,
    BaseLitReversiModel,
    QuantizationConfig,
    WeightClipConfig,
    screlu,
)
from models.stacked_linear import StackedLinear
from sparse_linear import SparseLinear

# Layer dimensions
LINPUT = 256

# Bucket configuration
NUM_LS_BUCKETS = 60
MAX_PLY = 60

# Quantization configuration for WASM model (8-bit weights)
WASM_MODEL_CONFIG = QuantizationConfig(
    score_scale=64.0,
    eval_score_scale=256.0,
    weight_scale_hidden=64.0,
    quantized_one=255.0,
    quantized_weight_max=127.0,
)


class LayerStacks(nn.Module):
    """Phase-bucketed output layer for ply-dependent processing."""

    def __init__(self, count: int) -> None:
        super().__init__()
        self.count = count
        self.bucket_size = MAX_PLY // count

        # Use StackedLinear for output layer
        self.output = StackedLinear(LINPUT, 1, count)

        # Zero output bias for stable initialization
        with torch.no_grad():
            self.output.linear.bias.zero_()

    def forward(self, x_pa: torch.Tensor, ply: torch.Tensor) -> torch.Tensor:
        ls_indices = ply.view(-1) // self.bucket_size
        return self.output(x_pa, ls_indices)

    def get_layer_stacks(self) -> Iterator[nn.Linear]:
        for i in range(self.count):
            yield self.output.at_index(i)


class ReversiWasmModel(nn.Module):
    """WASM-compatible Reversi evaluation with 8-bit weight quantization."""

    def __init__(self, config: QuantizationConfig = WASM_MODEL_CONFIG) -> None:
        super().__init__()
        self.config = config

        # Legacy attribute access for backward compatibility
        self.score_scale = config.score_scale
        self.eval_score_scale = config.eval_score_scale
        self.weight_scale_out = config.eval_score_scale * 128.0  # WASM-specific
        self.quantized_one = config.quantized_one

        self.max_input_weight = config.quantized_weight_max / config.quantized_one

        # Feature offset buffer for converting raw indices to absolute indices
        self.register_buffer(
            "feature_offsets",
            torch.tensor(FEATURE_CUM_OFFSETS, dtype=torch.int64),
            persistent=False,
        )

        self.input = SparseLinear(SUM_OF_FEATURES, LINPUT)
        self.layer_stacks = LayerStacks(NUM_LS_BUCKETS)

    def forward(
        self,
        feature_indices: torch.Tensor,
        mobility: torch.Tensor,  # unused, for API compatibility
        ply: torch.Tensor,
    ) -> torch.Tensor:
        feature_indices = feature_indices + self.feature_offsets

        x = self.input(feature_indices)
        x = screlu(x, self.config.activation_scale)
        return self.layer_stacks(x, ply)


class LitReversiWasmModel(BaseLitReversiModel):
    """Lightning wrapper for ReversiWasmModel."""

    def __init__(
        self,
        lr: float = 0.001,
        weight_decay: float = 1e-2,
        t_max: int = 100,
        eta_min: float = 1e-8,
    ) -> None:
        super().__init__(lr=lr, weight_decay=weight_decay, t_max=t_max, eta_min=eta_min)

        self.model = ReversiWasmModel()
        self.weight_clipping = [
            WeightClipConfig(
                params=[self.model.input.weight],
                min_weight=-self.model.max_input_weight,
                max_weight=self.model.max_input_weight,
            ),
        ]

    def forward(
        self,
        feature_indices: torch.Tensor,
        mobility: torch.Tensor,
        ply: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(feature_indices, mobility, ply)
