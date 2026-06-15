"""WASM-compatible Reversi neural network model."""

from __future__ import annotations

from dataclasses import dataclass
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
LHIDDEN = 16
LOUTPUT = LINPUT + LHIDDEN

# Bucket configuration
NUM_LS_BUCKETS = 60
MAX_PLY = 60


@dataclass(frozen=True)
class WasmQuantizationConfig(QuantizationConfig):
    """Quantization parameters for the WASM model."""

    input_quantized_one: float = 255.0
    hidden_quantized_one: float = 1023.0

    @property
    def input_activation_scale(self) -> float:
        return self.input_quantized_one / (self.input_quantized_one + 1)

    @property
    def hidden_activation_scale(self) -> float:
        return self.hidden_quantized_one / (self.hidden_quantized_one + 1)

    @property
    def activation_scale(self) -> float:
        return self.hidden_activation_scale


# Quantization configuration for WASM model
WASM_MODEL_CONFIG = WasmQuantizationConfig(
    score_scale=64.0,
    eval_score_scale=256.0,
    weight_scale_hidden=1024.0,
    weight_scale_out=32768.0,  # eval_score_scale * 128.0
    quantized_one=255.0,
    input_quantized_one=255.0,
    hidden_quantized_one=1023.0,
    quantized_weight_max=127.0,
)


class LayerStacks(nn.Module):
    """Phase-bucketed layers with skip connection to input features."""

    def __init__(self, count: int, activation_scale: float) -> None:
        super().__init__()
        self.count = count
        self.bucket_size = MAX_PLY // count
        self.activation_scale = activation_scale

        self.l1 = StackedLinear(LINPUT, LHIDDEN, count)
        self.l2 = StackedLinear(LHIDDEN, LHIDDEN, count)
        self.output = StackedLinear(LOUTPUT, 1, count)

        # Zero output bias for stable initialization
        with torch.no_grad():
            self.output.linear.bias.zero_()

    def forward(self, x: torch.Tensor, ply: torch.Tensor) -> torch.Tensor:
        ls_indices = ply.view(-1) // self.bucket_size

        l1x = self.l1(x, ls_indices)
        l1x = screlu(l1x, self.activation_scale)

        l2x = self.l2(l1x, ls_indices)
        l2x = screlu(l2x, self.activation_scale)

        output_features = torch.cat([l2x, x], dim=1)
        return self.output(output_features, ls_indices)

    def get_layer_stacks(self) -> Iterator[tuple[nn.Linear, nn.Linear, nn.Linear]]:
        for i in range(self.count):
            yield (
                self.l1.at_index(i),
                self.l2.at_index(i),
                self.output.at_index(i),
            )


class ReversiWasmModel(nn.Module):
    """WASM-compatible Reversi evaluation with mixed weight quantization."""

    def __init__(self, config: WasmQuantizationConfig = WASM_MODEL_CONFIG) -> None:
        super().__init__()
        self.config = config
        self.input_quantized_one = config.input_quantized_one
        self.hidden_quantized_one = config.hidden_quantized_one
        self.input_activation_scale = config.input_activation_scale
        self.hidden_activation_scale = config.hidden_activation_scale
        self.max_input_weight = config.quantized_weight_max / config.input_quantized_one

        # Feature offset buffer for converting raw indices to absolute indices
        self.register_buffer(
            "feature_offsets",
            torch.tensor(FEATURE_CUM_OFFSETS, dtype=torch.int64),
            persistent=False,
        )

        self.input = SparseLinear(SUM_OF_FEATURES, LINPUT)
        self.layer_stacks = LayerStacks(NUM_LS_BUCKETS, config.hidden_activation_scale)

    def forward(
        self,
        feature_indices: torch.Tensor,
        mobility: torch.Tensor,  # unused, for API compatibility
        ply: torch.Tensor,
    ) -> torch.Tensor:
        feature_indices = feature_indices + self.feature_offsets

        x = self.input(feature_indices)
        x = screlu(x, self.input_activation_scale)
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
        hidden_weight_scale = self.model.config.weight_scale_hidden
        max_l1_weight = torch.iinfo(torch.int16).max / (
            hidden_weight_scale
            * self.model.hidden_quantized_one
            / self.model.input_quantized_one
        )
        max_l2_weight = torch.iinfo(torch.int16).max / hidden_weight_scale
        layer_stacks = self.model.layer_stacks
        self.weight_clipping = [
            WeightClipConfig(
                params=[self.model.input.weight],
                min_weight=-self.model.max_input_weight,
                max_weight=self.model.max_input_weight,
            ),
            WeightClipConfig(
                params=[
                    layer_stacks.l1.linear.weight,
                ],
                min_weight=-max_l1_weight,
                max_weight=max_l1_weight,
            ),
            WeightClipConfig(
                params=[
                    layer_stacks.l2.linear.weight,
                ],
                min_weight=-max_l2_weight,
                max_weight=max_l2_weight,
            ),
        ]

    def forward(
        self,
        feature_indices: torch.Tensor,
        mobility: torch.Tensor,
        ply: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(feature_indices, mobility, ply)
