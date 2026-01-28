"""Main Reversi neural network model for full-strength evaluation.

This module implements the primary neural network architecture for Reversi
position evaluation, featuring phase-adaptive input layers and layer stacks
for ply-dependent processing.
"""

from __future__ import annotations

from typing import Any, Iterator

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_common import (
    SUM_OF_FEATURES,
    QuantizationConfig,
    WeightClipConfig,
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
LB = 128
LOUTPUT = L3 + LPA + (LBASE // 2)

# Bucket configuration
NUM_PA_BUCKETS = 6
NUM_LS_BUCKETS = 60
MAX_PLY = 60

# Type alias for optimizer parameter groups
ParamGroup = dict[str, Any]


class LayerStacks(nn.Module):
    """Phase-bucketed layer stacks for efficient ply-dependent processing.

    Each layer stack processes features through L1 -> L2 -> output layers,
    with different weights for different game phases (determined by ply).

    Args:
        count: Number of layer stack buckets.
    """

    def __init__(self, count: int) -> None:
        super().__init__()
        self.count = count
        self.bucket_size = MAX_PLY // count

        # Use StackedLinear for all phase-bucketed layers
        self.l1_base = StackedLinear(L1_BASE, L2_HALF, count)
        self.l1_pa = StackedLinear(L1_PA, L2_HALF, count)
        self.l2 = StackedLinear(L2 * 2, L3, count)
        self.output = StackedLinear(LOUTPUT, 1, count)

        # Zero output bias for stable initialization
        with torch.no_grad():
            self.output.linear.bias.zero_()

    def forward(
        self,
        x_base: torch.Tensor,
        x_pa: torch.Tensor,
        mobility: torch.Tensor,
        ply: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through phase-selected layer stack.

        Args:
            x_base: Base feature activations (batch, LBASE // 2).
            x_pa: Phase-adaptive feature activations (batch, LPA).
            mobility: Mobility features (batch, 1).
            ply: Current ply for each sample (batch,).

        Returns:
            Evaluation scores (batch, 1).
        """
        # Compute bucket indices once
        ls_indices = ply.view(-1) // self.bucket_size

        # Add mobility features
        mobility_scaled = torch.clamp(mobility * (7.0 / 255.0), max=1.0)
        x_base_with_mobility = torch.cat([x_base, mobility_scaled], dim=1)
        x_pa_with_mobility = torch.cat([x_pa, mobility_scaled], dim=1)

        # Process base and PA features through L1
        l1x_base = self.l1_base(x_base_with_mobility, ls_indices)
        l1x_pa = self.l1_pa(x_pa_with_mobility, ls_indices)

        # Combine and apply activations
        l1x = torch.cat([l1x_base, l1x_pa], dim=1)
        l1x_squared = l1x.pow(2) * (255 / 256)
        l1x = torch.cat([l1x_squared, l1x], dim=1)
        l1x = torch.clamp(l1x, 0.0, 1.0)

        # Second layer
        l2x = self.l2(l1x, ls_indices)
        l2x = torch.clamp(l2x, 0.0, 1.0).pow(2) * (255 / 256)

        # Output layer
        output_features = torch.cat([l2x, x_base, x_pa], dim=1)
        output = self.output(output_features, ls_indices)

        return output

    def get_layer_stacks(
        self,
    ) -> Iterator[tuple[nn.Linear, nn.Linear, nn.Linear, nn.Linear]]:
        """Extract individual layer stacks as separate nn.Linear modules.

        Yields:
            Tuple of (l1_base, l1_pa, l2, output) Linear modules for each stack.
        """
        for i in range(self.count):
            yield (
                self.l1_base.at_index(i),
                self.l1_pa.at_index(i),
                self.l2.at_index(i),
                self.output.at_index(i),
            )


class ReversiModel(nn.Module):
    """Full-strength Reversi position evaluation model.

    This model uses a combination of base sparse features and phase-adaptive
    features, processed through layer stacks that vary with game phase.

    Attributes:
        config: Quantization configuration for model serialization.
    """

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

        # Networks
        self.base_input = SparseLinear(SUM_OF_FEATURES, LBASE)
        self.pa_input = PhaseAdaptiveInput(
            count=NUM_PA_BUCKETS,
            output_dim=LPA,
            max_ply=MAX_PLY,
            activation_scale=255 / 256,
        )
        self.layer_stacks = LayerStacks(NUM_LS_BUCKETS)

    def forward(
        self,
        feature_indices: torch.Tensor,
        mobility: torch.Tensor,
        ply: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate positions from sparse features.

        Args:
            feature_indices: Sparse feature indices (batch, num_features).
            mobility: Mobility features (batch, 1).
            ply: Current ply for each sample (batch,).

        Returns:
            Position evaluation scores (batch, 1).
        """
        # Base input processing
        x_base = self.base_input(feature_indices)
        x_base1, x_base2 = torch.split(x_base, LBASE // 2, dim=1)

        # Apply different activations to each half
        x_base1 = torch.clamp(x_base1, 0.0, 1.0)
        x_base2 = torch.clamp(x_base2, 0.0, 1.0)

        # Combine
        x_base = x_base1 * x_base2 * (255 / 256)

        # Phase-adaptive input
        x_pa = self.pa_input(feature_indices, ply)

        return self.layer_stacks(x_base, x_pa, mobility, ply)


class LitReversiModel(L.LightningModule):
    """PyTorch Lightning wrapper for ReversiModel training.

    Handles training loop, validation, optimizer configuration, and
    weight clipping for quantization-aware training.

    Args:
        lr: Learning rate.
        weight_decay: Weight decay for regularization.
        t_max: Maximum iterations for cosine annealing scheduler.
        eta_min: Minimum learning rate for scheduler.
    """

    def __init__(
        self,
        lr: float = 0.001,
        weight_decay: float = 1e-2,
        t_max: int = 100,
        eta_min: float = 1e-8,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Core model
        self.model = ReversiModel()

        # Weight clipping configuration for quantization-aware training
        self.weight_clipping = [
            WeightClipConfig(
                params=[self.model.layer_stacks.l1_base.linear.weight],
                min_weight=-self.model.max_hidden_weight,
                max_weight=self.model.max_hidden_weight,
            ),
            WeightClipConfig(
                params=[self.model.layer_stacks.l1_pa.linear.weight],
                min_weight=-self.model.max_hidden_weight,
                max_weight=self.model.max_hidden_weight,
            ),
            WeightClipConfig(
                params=[self.model.layer_stacks.l2.linear.weight],
                min_weight=-self.model.max_hidden_weight,
                max_weight=self.model.max_hidden_weight,
            ),
        ]

    def forward(
        self,
        feature_indices: torch.Tensor,
        mobility: torch.Tensor,
        ply: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass delegating to the core model."""
        return self.model(feature_indices, mobility, ply)

    @torch.compile(fullgraph=True, options={"shape_padding": True})
    def _step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Compute loss for a single batch."""
        score_target, feature_indices, mobility, ply = batch

        score_pred = self(feature_indices, mobility, ply)
        score_target_scaled = score_target / self.model.score_scale
        return F.mse_loss(score_pred, score_target_scaled)

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        """Execute one training step."""
        loss = self._step(batch, batch_idx)
        return {"loss": loss}

    def on_train_batch_end(
        self,
        outputs: dict[str, torch.Tensor],
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Log training metrics after each batch."""
        self.log_dict({"train_loss": outputs["loss"]})

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Execute one validation step."""
        loss = self._step(batch, batch_idx)
        self.log("val_loss", loss)

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
        optimizer_closure: Any,
    ) -> None:
        """Execute optimizer step with weight clipping."""
        optimizer.step(closure=optimizer_closure)

        # Clip weights for quantization
        with torch.no_grad():
            for clip_config in self.weight_clipping:
                for param in clip_config.params:
                    param.clamp_(clip_config.min_weight, clip_config.max_weight)

    def _build_param_groups(self) -> list[ParamGroup]:
        """Build parameter groups with different weight decay settings."""
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

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure AdamW optimizer with Lookahead and cosine annealing."""
        param_groups = self._build_param_groups()

        betas = (0.9, 0.999)
        eps = 1e-8

        try:
            import apex

            optimizer: torch.optim.Optimizer = apex.optimizers.FusedAdam(
                params=param_groups,
                lr=self.hparams.lr,
                betas=betas,
                eps=eps,
            )
        except ImportError:
            optimizer = torch.optim.AdamW(
                params=param_groups,
                lr=self.hparams.lr,
                betas=betas,
                eps=eps,
                fused=True,
            )

        import timm.optim

        optimizer = timm.optim.Lookahead(
            optimizer,
            alpha=0.5,
            k=6,
        )

        from torch.optim.lr_scheduler import CosineAnnealingLR

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.t_max,
            eta_min=self.hparams.eta_min,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
