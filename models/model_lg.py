"""Main Reversi neural network model for full-strength evaluation."""

from __future__ import annotations

from typing import Iterator

import torch
import torch.nn as nn

from models.model_common import (
    FEATURE_CUM_OFFSETS,
    SUM_OF_FEATURES,
    BaseLitReversiModel,
    ParamGroup,
    QuantizationConfig,
    WeightClipConfig,
    screlu,
)
from models.phase_adaptive_input import PhaseAdaptiveInput
from models.stacked_linear import StackedLinear
from sparse_linear import SparseLinear

# Layer dimensions
LBASE = 256
LPA = 128
L1_BASE = (LBASE // 2) + 1  # Base input to L1 (+1 for mobility)
L1_PA = LPA + 1  # Phase-adaptive input to L1 (+1 for mobility)
L2 = 16
L2_HALF = L2 // 2
L3 = 64
LOUTPUT = L3 + LPA + (LBASE // 2)

# Bucket configuration
NUM_PA_BUCKETS = 6
NUM_LS_BUCKETS = 60
MAX_PLY = 60


class LayerStacks(nn.Module):
    """Phase-bucketed layer stacks with skip connections.

    Architecture: inputs -> L1 -> L2 -> concat(L2, original_inputs) -> output
    All layers use ply-dependent weights selected by phase bucket.
    """

    def __init__(self, count: int, activation_scale: float) -> None:
        super().__init__()
        self.count = count
        self.bucket_size = MAX_PLY // count
        self.activation_scale = activation_scale

        # Use StackedLinear for all phase-bucketed layers
        self.l1_base = StackedLinear(L1_BASE, L2_HALF, count)
        self.l1_pa = StackedLinear(L1_PA, L2_HALF, count)
        self.l2 = StackedLinear(L2 * 2, L3, count)
        self.output = StackedLinear(LOUTPUT, 1, count)

        with torch.no_grad():
            self.output.linear.bias.zero_()

    def forward(
        self,
        x_base: torch.Tensor,
        x_pa: torch.Tensor,
        mobility: torch.Tensor,
        ply: torch.Tensor,
    ) -> torch.Tensor:
        ls_indices = ply.view(-1) // self.bucket_size

        mobility_scaled = torch.clamp(mobility * (7.0 / 255.0), max=1.0)
        x_base_with_mobility = torch.cat([x_base, mobility_scaled], dim=1)
        x_pa_with_mobility = torch.cat([x_pa, mobility_scaled], dim=1)

        l1x_base = self.l1_base(x_base_with_mobility, ls_indices)
        l1x_pa = self.l1_pa(x_pa_with_mobility, ls_indices)

        l1x = torch.cat([l1x_base, l1x_pa], dim=1)
        l1x_squared = l1x.pow(2) * self.activation_scale
        l1x = torch.cat([l1x_squared, l1x], dim=1)
        l1x = torch.clamp(l1x, 0.0, 1.0)

        l2x = self.l2(l1x, ls_indices)
        l2x = screlu(l2x, self.activation_scale)

        output_features = torch.cat([l2x, x_base, x_pa], dim=1)
        output = self.output(output_features, ls_indices)

        return output

    def get_layer_stacks(
        self,
    ) -> Iterator[tuple[nn.Linear, nn.Linear, nn.Linear, nn.Linear]]:
        """Yield (l1_base, l1_pa, l2, output) for each stack."""
        for i in range(self.count):
            yield (
                self.l1_base.at_index(i),
                self.l1_pa.at_index(i),
                self.l2.at_index(i),
                self.output.at_index(i),
            )


class ReversiModel(nn.Module):
    """Full-strength Reversi evaluation with base sparse + phase-adaptive features."""

    def __init__(self, config: QuantizationConfig = QuantizationConfig()) -> None:
        super().__init__()
        self.config = config

        # Legacy attribute access for backward compatibility
        self.score_scale = config.score_scale
        self.eval_score_scale = config.eval_score_scale
        self.weight_scale_hidden = config.weight_scale_hidden
        self.weight_scale_out = config.weight_scale_out
        self.quantized_one = config.quantized_one
        self.quantized_weight_max = config.quantized_weight_max
        self.max_hidden_weight = config.max_hidden_weight

        self.register_buffer(
            "feature_offsets",
            torch.tensor(FEATURE_CUM_OFFSETS, dtype=torch.int64),
            persistent=False,
        )

        # Networks
        self.base_input = SparseLinear(SUM_OF_FEATURES, LBASE)
        self.pa_input = PhaseAdaptiveInput(
            count=NUM_PA_BUCKETS,
            output_dim=LPA,
            max_ply=MAX_PLY,
            activation_scale=config.activation_scale,
        )
        self.layer_stacks = LayerStacks(NUM_LS_BUCKETS, config.activation_scale)

    def forward(
        self,
        feature_indices: torch.Tensor,
        mobility: torch.Tensor,
        ply: torch.Tensor,
    ) -> torch.Tensor:
        feature_indices = feature_indices + self.feature_offsets

        x_base = self.base_input(feature_indices)
        x_base1, x_base2 = torch.split(x_base, LBASE // 2, dim=1)

        x_base1 = torch.clamp(x_base1, 0.0, 1.0)
        x_base2 = torch.clamp(x_base2, 0.0, 1.0)
        x_base = x_base1 * x_base2 * self.config.activation_scale

        x_pa = self.pa_input(feature_indices, ply)

        return self.layer_stacks(x_base, x_pa, mobility, ply)


class LitReversiModel(BaseLitReversiModel):
    """Lightning wrapper for ReversiModel."""

    def __init__(
        self,
        lr: float = 0.001,
        weight_decay: float = 1e-2,
        t_max: int = 100,
        eta_min: float = 1e-8,
    ) -> None:
        super().__init__(lr=lr, weight_decay=weight_decay, t_max=t_max, eta_min=eta_min)

        self.model = ReversiModel()
        max_weight = self.model.max_hidden_weight
        layer_stacks = self.model.layer_stacks
        self.weight_clipping = [
            WeightClipConfig(
                params=[
                    layer_stacks.l1_base.linear.weight,
                    layer_stacks.l1_pa.linear.weight,
                    layer_stacks.l2.linear.weight,
                ],
                min_weight=-max_weight,
                max_weight=max_weight,
            ),
        ]

    def forward(
        self,
        feature_indices: torch.Tensor,
        mobility: torch.Tensor,
        ply: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(feature_indices, mobility, ply)

    def _build_param_groups(self) -> list[ParamGroup]:
        """Separate sparse weights from other parameters."""
        sparse_weight_suffixes = {"base_input.weight", "pa_input.input.weight"}
        param_buckets: dict[str, list[nn.Parameter]] = {
            "sparse": [],
            "decay": [],
            "no_decay": [],
        }

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith(".bias") or param.dim() == 1:
                param_buckets["no_decay"].append(param)
            elif any(name.endswith(suffix) for suffix in sparse_weight_suffixes):
                param_buckets["sparse"].append(param)
            else:
                param_buckets["decay"].append(param)

        weight_decay = self.hparams.weight_decay
        return [
            {"params": param_buckets["sparse"], "weight_decay": weight_decay},
            {"params": param_buckets["decay"], "weight_decay": weight_decay},
            {"params": param_buckets["no_decay"], "weight_decay": 0.0},
        ]
