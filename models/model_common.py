"""Shared utilities and configurations for Reversi neural network models.

This module provides common feature configurations, dataclasses for model settings,
and utility functions used across different model variants.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch.nn as nn

# Feature configuration: number of parameters per feature pattern.
# First 20 patterns use 3^8 = 6561 (8-cell patterns)
# Last 4 patterns use 3^9 = 19683 (9-cell patterns)
NUM_FEATURE_PARAMS: list[int] = [
    6561,
    6561,
    6561,
    6561,
    6561,
    6561,
    6561,
    6561,
    6561,
    6561,
    6561,
    6561,
    6561,
    6561,
    6561,
    6561,
    6561,
    6561,
    6561,
    6561,
    19683,
    19683,
    19683,
    19683,
]

NUM_FEATURES = len(NUM_FEATURE_PARAMS)
SUM_OF_FEATURES = sum(NUM_FEATURE_PARAMS)


@dataclass(frozen=True)
class QuantizationConfig:
    """Quantization parameters for model serialization.

    These parameters control how floating-point weights and biases are
    quantized to integers for deployment in the game engine.

    Attributes:
        score_scale: Scale factor for training target scores.
        eval_score_scale: Scale factor for evaluation scores in inference.
        weight_scale_hidden: Scale factor for hidden layer weights.
        quantized_one: Maximum value representing 1.0 after quantization.
        quantized_weight_max: Maximum absolute value for quantized weights.
    """

    score_scale: float = 64.0
    eval_score_scale: float = 256.0
    weight_scale_hidden: float = 64.0
    quantized_one: float = 255.0
    quantized_weight_max: float = 127.0

    @property
    def weight_scale_out(self) -> float:
        """Compute output layer weight scale from component scales."""
        return self.eval_score_scale * self.weight_scale_hidden

    @property
    def max_hidden_weight(self) -> float:
        """Compute maximum allowed hidden weight value before clipping."""
        return self.quantized_weight_max / self.weight_scale_hidden


@dataclass
class WeightClipConfig:
    """Weight clipping configuration for a parameter group.

    Used to constrain weights during training to ensure they remain
    within quantizable ranges.

    Attributes:
        params: List of parameters to clip.
        min_weight: Minimum allowed weight value.
        max_weight: Maximum allowed weight value.
    """

    params: list[nn.Parameter] = field(default_factory=list)
    min_weight: float = 0.0
    max_weight: float = 0.0
