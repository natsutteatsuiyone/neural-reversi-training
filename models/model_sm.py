"""Small Reversi neural network model for lightweight evaluation."""

from __future__ import annotations

from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_common import (
    FEATURE_CUM_OFFSETS,
    BaseLitReversiModel,
    QuantizationConfig,
)
from models.phase_adaptive_input import PhaseAdaptiveInput
from models.stacked_linear import StackedLinear

# Layer dimensions
LPA = 128
LOUTPUT = LPA

# Bucket configuration (small model uses 30-ply range)
NUM_PA_BUCKETS = 3
NUM_LS_BUCKETS = 30
MAX_PLY = 30

# Quantization configuration for small model
SMALL_MODEL_CONFIG = QuantizationConfig(
    score_scale=64.0,
    eval_score_scale=256.0,
    weight_scale_hidden=64.0,
    weight_scale_out=65536.0,  # eval_score_scale * 256.0
    quantized_one=1023.0,  # 10-bit quantization
    quantized_weight_max=127.0,
)


class LayerStacks(nn.Module):
    """Phase-bucketed output layer for ply-dependent processing."""

    def __init__(self, count: int) -> None:
        super().__init__()
        self.count = count
        self.bucket_size = MAX_PLY // count

        # Use StackedLinear for output layer
        self.output = StackedLinear(LOUTPUT, 1, count)

        # Zero output bias for stable initialization
        with torch.no_grad():
            self.output.linear.bias.zero_()

    def forward(self, x: torch.Tensor, ply: torch.Tensor) -> torch.Tensor:
        ls_indices = ply.view(-1) // self.bucket_size
        return self.output(x, ls_indices)

    def get_layer_stacks(self) -> Iterator[nn.Linear]:
        for i in range(self.count):
            yield self.output.at_index(i)


class ReversiSmallModel(nn.Module):
    """Lightweight Reversi evaluation with phase-adaptive features only."""

    def __init__(self, config: QuantizationConfig = SMALL_MODEL_CONFIG) -> None:
        super().__init__()
        self.config = config

        # Legacy attribute access for backward compatibility
        self.score_scale = config.score_scale
        self.eval_score_scale = config.eval_score_scale
        self.weight_scale_hidden = config.weight_scale_hidden
        self.weight_scale_out = config.weight_scale_out
        self.quantized_one = config.quantized_one

        # Feature offset buffer for converting raw indices to absolute indices
        self.register_buffer(
            "feature_offsets",
            torch.tensor(FEATURE_CUM_OFFSETS, dtype=torch.int64),
            persistent=False,
        )

        self.pa_input = PhaseAdaptiveInput(
            count=NUM_PA_BUCKETS,
            output_dim=LPA,
            max_ply=MAX_PLY,
            activation_scale=config.activation_scale,
        )
        self.layer_stacks = LayerStacks(NUM_LS_BUCKETS)

    def forward(
        self,
        feature_indices: torch.Tensor,
        ply: torch.Tensor,
    ) -> torch.Tensor:
        feature_indices = feature_indices + self.feature_offsets

        x_pa = self.pa_input(feature_indices, ply)
        return self.layer_stacks(x_pa, ply)


class LitReversiSmallModel(BaseLitReversiModel):
    """Lightning wrapper for ReversiSmallModel."""

    def __init__(
        self,
        lr: float = 0.001,
        weight_decay: float = 1e-2,
        t_max: int = 100,
        eta_min: float = 1e-8,
    ) -> None:
        super().__init__(lr=lr, weight_decay=weight_decay, t_max=t_max, eta_min=eta_min)
        self.model = ReversiSmallModel()

    def forward(
        self,
        feature_indices: torch.Tensor,
        ply: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(feature_indices, ply)

    @torch.compile(fullgraph=True, options={"shape_padding": True})
    def _step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Compute loss for a single batch (small model ignores mobility, offsets ply)."""
        score_target, feature_indices, _mobility, ply = batch
        ply = ply.sub(30)
        score_pred = self(feature_indices, ply)
        return F.mse_loss(score_pred, score_target / self.model.score_scale)
