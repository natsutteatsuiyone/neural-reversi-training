"""Small Reversi neural network model for lightweight evaluation.

This module implements a smaller, faster neural network architecture
optimized for resource-constrained environments while maintaining
reasonable evaluation quality.
"""

from __future__ import annotations

from typing import Any, Iterator

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_common import QuantizationConfig
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
    quantized_one=1023.0,  # 10-bit quantization
    quantized_weight_max=127.0,
)


class LayerStacks(nn.Module):
    """Phase-bucketed output layer for ply-dependent processing.

    Simplified layer stack with only output layer, suitable for
    the small model architecture.

    Args:
        count: Number of layer stack buckets.
    """

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
        """Forward pass through phase-selected output layer.

        Args:
            x: Phase-adaptive feature activations (batch, LOUTPUT).
            ply: Current ply for each sample (batch,).

        Returns:
            Evaluation scores (batch, 1).
        """
        ls_indices = ply.view(-1) // self.bucket_size
        return self.output(x, ls_indices)

    def get_layer_stacks(self) -> Iterator[nn.Linear]:
        """Extract individual layer stacks as separate nn.Linear modules.

        Yields:
            Output nn.Linear module for each stack.
        """
        for i in range(self.count):
            yield self.output.at_index(i)


class ReversiSmallModel(nn.Module):
    """Small Reversi position evaluation model.

    A lightweight model with only phase-adaptive features, suitable
    for resource-constrained environments.

    Attributes:
        config: Quantization configuration for model serialization.
    """

    def __init__(self, config: QuantizationConfig = SMALL_MODEL_CONFIG) -> None:
        super().__init__()
        self.config = config

        # Legacy attribute access for backward compatibility
        self.score_scale = config.score_scale
        self.eval_score_scale = config.eval_score_scale
        self.weight_scale_hidden = config.weight_scale_hidden
        self.weight_scale_out = config.weight_scale_out
        self.quantized_one = config.quantized_one

        self.pa_input = PhaseAdaptiveInput(
            count=NUM_PA_BUCKETS,
            output_dim=LPA,
            max_ply=MAX_PLY,
            activation_scale=1023 / 1024,
        )
        self.layer_stacks = LayerStacks(NUM_LS_BUCKETS)

    def forward(
        self,
        feature_indices: torch.Tensor,
        ply: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate positions from sparse features.

        Args:
            feature_indices: Sparse feature indices (batch, num_features).
            ply: Current ply for each sample (batch,).

        Returns:
            Position evaluation scores (batch, 1).
        """
        x_pa = self.pa_input(feature_indices, ply)
        return self.layer_stacks(x_pa, ply)


class LitReversiSmallModel(L.LightningModule):
    """PyTorch Lightning wrapper for ReversiSmallModel training.

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
        eta_min: float = 1e-9,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = ReversiSmallModel()

    def forward(
        self,
        feature_indices: torch.Tensor,
        ply: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass delegating to the core model."""
        return self.model(feature_indices, ply)

    @torch.compile(fullgraph=True, options={"shape_padding": True})
    def _step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Compute loss for a single batch."""
        score_target, feature_indices, _mobility, ply = batch
        ply = ply.sub(30)

        score_pred = self(feature_indices, ply)
        return F.mse_loss(score_pred, score_target / self.model.score_scale)

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

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure AdamW optimizer with Lookahead and cosine annealing."""
        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith(".bias") or param.dim() == 1:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        params = [
            {"params": decay_params, "weight_decay": self.hparams.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        betas = (0.9, 0.999)
        eps = 1e-8
        try:
            import apex

            optimizer: torch.optim.Optimizer = apex.optimizers.FusedAdam(
                params=params,
                lr=self.hparams.lr,
                betas=betas,
                eps=eps,
            )
        except ImportError:
            optimizer = torch.optim.AdamW(
                params=params,
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
