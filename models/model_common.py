"""Shared utilities and configurations for Reversi neural network models.

This module provides common feature configurations, dataclasses for model settings,
and utility functions used across different model variants.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from itertools import accumulate
from typing import Any

import lightning as L
import timm.optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

# Feature configuration: number of parameters per feature pattern.
# First 20 patterns use 3^8 = 6561 (8-cell patterns)
# Last 4 patterns use 3^9 = 19683 (9-cell patterns)
NUM_FEATURE_PARAMS: tuple[int, ...] = (
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
)

NUM_FEATURES = len(NUM_FEATURE_PARAMS)
SUM_OF_FEATURES = sum(NUM_FEATURE_PARAMS)

# Cumulative offsets for feature indices: FEATURE_CUM_OFFSETS[i] = sum(NUM_FEATURE_PARAMS[:i])
FEATURE_CUM_OFFSETS: tuple[int, ...] = tuple(accumulate((0,) + NUM_FEATURE_PARAMS[:-1]))


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
    """Weight clipping bounds for quantization-aware training."""

    params: list[nn.Parameter] = field(default_factory=list)
    min_weight: float = 0.0
    max_weight: float = 0.0


ParamGroup = dict[str, Any]


def create_optimizer_with_scheduler(
    param_groups: list[ParamGroup],
    lr: float,
    t_max: int,
    eta_min: float,
) -> dict[str, Any]:
    """Create Adam-based optimizer with Lookahead and cosine annealing.

    Uses Apex FusedAdam if available, otherwise PyTorch AdamW.
    """
    betas = (0.9, 0.999)
    eps = 1e-8

    try:
        import apex

        optimizer: torch.optim.Optimizer = apex.optimizers.FusedAdam(
            params=param_groups,
            lr=lr,
            betas=betas,
            eps=eps,
        )
    except ImportError:
        warnings.warn(
            "Apex not available, falling back to PyTorch AdamW optimizer. "
            "Install NVIDIA Apex for potentially improved training performance.",
            UserWarning,
            stacklevel=2,
        )
        optimizer = torch.optim.AdamW(
            params=param_groups,
            lr=lr,
            betas=betas,
            eps=eps,
            fused=True,
        )

    optimizer = timm.optim.Lookahead(optimizer, alpha=0.5, k=6)
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)

    return {
        "optimizer": optimizer,
        "lr_scheduler": scheduler,
        "monitor": "val_loss",
    }


def build_simple_param_groups(
    module: L.LightningModule,
    weight_decay: float,
) -> list[ParamGroup]:
    """Build parameter groups: biases and 1D params get no weight decay."""
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []

    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or param.dim() == 1:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


class BaseLitReversiModel(L.LightningModule):
    """Base Lightning module with shared training logic.

    Subclasses must define:
        - model: The core nn.Module with a score_scale attribute
        - forward(): Delegating to self.model

    Optional overrides:
        - weight_clipping: List of WeightClipConfig for quantization-aware training
        - _build_param_groups(): Custom parameter grouping for optimizer
    """

    model: nn.Module
    weight_clipping: list[WeightClipConfig]

    def __init__(
        self,
        lr: float = 0.001,
        weight_decay: float = 1e-2,
        t_max: int = 100,
        eta_min: float = 1e-8,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.weight_clipping = []

    @torch.compile(fullgraph=True, options={"shape_padding": True})
    def _step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Compute loss for a single batch. Override if batch unpacking differs."""
        score_target, feature_indices, mobility, ply = batch
        score_pred = self(feature_indices, mobility, ply)
        return F.mse_loss(score_pred, score_target / self.model.score_scale)

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        return {"loss": self._step(batch, batch_idx)}

    def on_train_batch_end(
        self,
        outputs: dict[str, torch.Tensor],
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        self.log_dict({"train_loss": outputs["loss"]})

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        self.log("val_loss", self._step(batch, batch_idx))

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
        optimizer_closure: Any,
    ) -> None:
        optimizer.step(closure=optimizer_closure)
        if self.weight_clipping:
            with torch.no_grad():
                for clip_config in self.weight_clipping:
                    for param in clip_config.params:
                        param.clamp_(clip_config.min_weight, clip_config.max_weight)

    def _build_param_groups(self) -> list[ParamGroup]:
        return build_simple_param_groups(self, self.hparams.weight_decay)

    def configure_optimizers(self) -> dict[str, Any]:
        return create_optimizer_with_scheduler(
            param_groups=self._build_param_groups(),
            lr=self.hparams.lr,
            t_max=self.hparams.t_max,
            eta_min=self.hparams.eta_min,
        )
