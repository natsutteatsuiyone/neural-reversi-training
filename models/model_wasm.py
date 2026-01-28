"""WASM-compatible Reversi neural network model.

This module implements a neural network architecture optimized for
WebAssembly deployment, with constraints suitable for browser-based
game engines.
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
    """Phase-bucketed output layer for ply-dependent processing.

    Args:
        count: Number of layer stack buckets.
    """

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
        """Forward pass through phase-selected output layer.

        Args:
            x_pa: Input feature activations (batch, LINPUT).
            ply: Current ply for each sample (batch,).

        Returns:
            Evaluation scores (batch, 1).
        """
        ls_indices = ply.view(-1) // self.bucket_size
        return self.output(x_pa, ls_indices)

    def get_layer_stacks(self) -> Iterator[nn.Linear]:
        """Extract individual layer stacks as separate nn.Linear modules.

        Yields:
            Output nn.Linear module for each stack.
        """
        for i in range(self.count):
            yield self.output.at_index(i)


class ReversiWasmModel(nn.Module):
    """WASM-compatible Reversi position evaluation model.

    Optimized for WebAssembly deployment with 8-bit weight quantization.

    Attributes:
        config: Quantization configuration for model serialization.
    """

    def __init__(self, config: QuantizationConfig = WASM_MODEL_CONFIG) -> None:
        super().__init__()
        self.config = config

        # Legacy attribute access for backward compatibility
        self.score_scale = config.score_scale
        self.eval_score_scale = config.eval_score_scale
        self.weight_scale_out = config.eval_score_scale * 128.0  # WASM-specific
        self.quantized_one = config.quantized_one

        self.max_input_weight = 127.0 / config.quantized_one

        self.input = SparseLinear(SUM_OF_FEATURES, LINPUT)
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
            mobility: Mobility features (unused, for API compatibility).
            ply: Current ply for each sample (batch,).

        Returns:
            Position evaluation scores (batch, 1).
        """
        x = self.input(feature_indices)
        x = x.clamp(0.0, 1.0).pow(2.0) * (255.0 / 256.0)
        return self.layer_stacks(x, ply)


class LitReversiWasmModel(L.LightningModule):
    """PyTorch Lightning wrapper for ReversiWasmModel training.

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

        self.model = ReversiWasmModel()

        # Weight clipping for quantization-aware training
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
