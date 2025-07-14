import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from typing import Tuple, List, Iterator

from quant import FakeQuantizeLinear, FakeQuantizeSparseLinear, fq_floor

NUM_FEATURE_PARAMS = [
    6561, 6561, 6561, 6561,
    6561, 6561,
    6561, 6561, 6561, 6561,
    6561, 6561, 6561, 6561,
    6561, 6561, 6561, 6561,
    19683, 19683, 19683, 19683,
]

NUM_FEATURES = len(NUM_FEATURE_PARAMS)
SUM_OF_FEATURES = sum(NUM_FEATURE_PARAMS)


LBASE = 192
LPA = 96
L1_BASE = (LBASE // 2) + 1
L1_PA = LPA + 1
L2 = 16
L2_HALF = L2 // 2
L3 = 64

NUM_PA_BUCKETS = 6
NUM_LS_BUCKETS = 60
MAX_PLY = 60


class LayerStacks(nn.Module):
    def __init__(
        self,
        count: int,
        quantized_one: float,
        weight_scale_hidden: float,
        weight_scale_out: float,
        score_scale: float,
    ):
        super().__init__()
        self.count = count
        self.bucket_size = MAX_PLY // count

        self.quantized_one = quantized_one
        self.weight_scale_hidden = weight_scale_hidden
        self.weight_scale_out = weight_scale_out
        self.score_scale = score_scale

        self.max_hidden_weight = self.quantized_one / self.weight_scale_hidden
        self.max_out_weight = (self.quantized_one * self.quantized_one) / (
            self.score_scale * self.weight_scale_out
        )

        self.l1_base = FakeQuantizeLinear(
            L1_BASE,
            L2_HALF * count,
            weight_scale=self.weight_scale_hidden,
            bias_scale=self.weight_scale_hidden * self.quantized_one,
        )
        self.l1_pa = FakeQuantizeLinear(
            L1_PA,
            L2_HALF * count,
            weight_scale=self.weight_scale_hidden,
            bias_scale=self.weight_scale_hidden * self.quantized_one,
        )
        self.l2 = FakeQuantizeLinear(
            L2 * 2,
            L3 * count,
            weight_scale=self.weight_scale_hidden,
            bias_scale=self.weight_scale_hidden * self.quantized_one,
        )
        self.output = FakeQuantizeLinear(
            L3,
            1 * count,
            weight_scale=self.score_scale * self.weight_scale_out / self.quantized_one,
            bias_scale=self.score_scale * self.weight_scale_out,
        )

        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize layers to ensure all layer stacks are initialized identically."""
        l1_base_weight = self.l1_base.weight
        l1_base_bias = self.l1_base.bias
        l1_pa_weight = self.l1_pa.weight
        l1_pa_bias = self.l1_pa.bias
        l2_weight = self.l2.weight
        l2_bias = self.l2.bias
        output_weight = self.output.weight
        output_bias = self.output.bias

        with torch.no_grad():
            output_bias.fill_(0.0)

            for i in range(1, self.count):
                start, end = i * L2_HALF, (i + 1) * L2_HALF
                l1_base_weight[start:end] = l1_base_weight[:L2_HALF]
                l1_base_bias[start:end] = l1_base_bias[:L2_HALF]
                l1_pa_weight[start:end] = l1_pa_weight[:L2_HALF]
                l1_pa_bias[start:end] = l1_pa_bias[:L2_HALF]

                start, end = i * L3, (i + 1) * L3
                l2_weight[start:end] = l2_weight[:L3]
                l2_bias[start:end] = l2_bias[:L3]

                output_weight[i:i+1] = output_weight[:1]
                output_bias[i:i+1] = output_bias[:1]

    def forward(self, x_base: torch.Tensor, x_pa: torch.Tensor, ply: torch.Tensor) -> torch.Tensor:
        batch_size = x_base.shape[0]
        idx_offset = torch.arange(0, batch_size * self.count, self.count, device=ply.device)
        ls_indices = ply.flatten() // self.bucket_size
        bucket_idx = (ls_indices + idx_offset).unsqueeze(1)

        # Process base features
        l1x_base = self.l1_base(x_base).view(-1, self.count, L2_HALF)
        l1x_base = torch.gather(l1x_base.view(-1, L2_HALF), 0, bucket_idx.expand(-1, L2_HALF))

        # Process PA features
        l1x_pa = self.l1_pa(x_pa).view(-1, self.count, L2_HALF)
        l1x_pa = torch.gather(l1x_pa.view(-1, L2_HALF), 0, bucket_idx.expand(-1, L2_HALF))

        # Combine and apply activations
        l1x = torch.cat([l1x_base, l1x_pa], dim=1)
        l1x_squared = l1x.pow(2) * (127 / 128)
        l1x = torch.cat([l1x_squared, l1x], dim=1).clamp(0.0, 1.0)
        l1x = fq_floor(l1x, self.quantized_one)

        # Second layer
        l2x = self.l2(l1x).view(-1, self.count, L3)
        l2x = torch.gather(l2x.view(-1, L3), 0, bucket_idx.expand(-1, L3))
        l2x = l2x.clamp(0.0, 1.0)
        l2x = fq_floor(l2x, self.quantized_one)

        # Output layer
        output = self.output(l2x).view(-1, self.count, 1)
        output = torch.gather(output.view(-1, 1), 0, bucket_idx)
        return fq_floor(output, self.score_scale)

    def get_coalesced_layer_stacks(self) -> Iterator[Tuple[nn.Linear, nn.Linear, nn.Linear, nn.Linear]]:
        """Extract individual layer stacks as separate nn.Linear modules."""
        for i in range(self.count):
            with torch.no_grad():
                l1_base = nn.Linear(L1_BASE, L2_HALF)
                l1_pa = nn.Linear(L1_PA, L2_HALF)
                l2 = nn.Linear(L2 * 2, L3)
                output = nn.Linear(L3, 1)

                # Extract weights and biases for this stack
                start_l2 = i * L2_HALF
                end_l2 = (i + 1) * L2_HALF
                start_l3 = i * L3
                end_l3 = (i + 1) * L3

                l1_base.weight.data = self.l1_base.weight[start_l2:end_l2]
                l1_base.bias.data = self.l1_base.bias[start_l2:end_l2]
                l1_pa.weight.data = self.l1_pa.weight[start_l2:end_l2]
                l1_pa.bias.data = self.l1_pa.bias[start_l2:end_l2]
                l2.weight.data = self.l2.weight[start_l3:end_l3]
                l2.bias.data = self.l2.bias[start_l3:end_l3]
                output.weight.data = self.output.weight[i:i+1]
                output.bias.data = self.output.bias[i:i+1]

                yield l1_base, l1_pa, l2, output


class PhaseAdaptiveInput(nn.Module):
    def __init__(self, count: int, quantized_one: float):
        super().__init__()
        self.quantized_one = quantized_one
        self.count = count
        self.bucket_size = MAX_PLY // self.count

        self.input = FakeQuantizeSparseLinear(
            SUM_OF_FEATURES,
            LPA * self.count,
            weight_scale=quantized_one,
            bias_scale=quantized_one,
        )
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize layers to ensure all phase-adaptive buckets are initialized identically."""
        with torch.no_grad():
            for i in range(1, self.count):
                start, end = i * LPA, (i + 1) * LPA
                self.input.weight[:, start:end] = self.input.weight[:, :LPA]
                self.input.bias[start:end] = self.input.bias[:LPA]

    def forward(
        self,
        feature_indices: torch.Tensor,
        values: torch.Tensor,
        batch_size: int,
        in_features: int,
        ply: torch.Tensor
    ) -> torch.Tensor:
        x = self.input(feature_indices, values, batch_size, in_features).view(-1, self.count, LPA)

        idx_offset = torch.arange(0, batch_size * self.count, self.count, device=ply.device)
        bucket_idx = (ply.flatten() // self.bucket_size + idx_offset).unsqueeze(1)
        x = torch.gather(x.view(-1, LPA), 0, bucket_idx.expand(-1, LPA))

        # Apply activation and normalization
        x = F.leaky_relu(x, negative_slope=0.125)
        x = x.clamp(-16/127, 1.0 - 16/127) + 16/127
        return fq_floor(x, self.quantized_one, 0.0, 1.0)

    def get_layers(self) -> Iterator[nn.Linear]:
        """Extract individual phase-adaptive layers as separate nn.Linear modules."""
        for i in range(self.count):
            with torch.no_grad():
                layer = nn.Linear(SUM_OF_FEATURES, LPA)
                start, end = i * LPA, (i + 1) * LPA
                layer.weight.data = self.input.weight[:, start:end]
                layer.bias.data = self.input.bias[start:end]
                yield layer


class ReversiModel(L.LightningModule):
    def __init__(
        self,
        lr: float = 0.001,
        weight_decay: float = 1e-2,
        t_max: int = 100,
        eta_min: float = 1e-10,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Quantization constants
        self.score_scale = 128.0
        self.weight_scale_hidden = 64.0
        self.weight_scale_out = 16.0
        self.quantized_one = 127.0

        # Networks
        self.base_input = FakeQuantizeSparseLinear(
            SUM_OF_FEATURES,
            LBASE,
            weight_scale=self.quantized_one,
            bias_scale=self.quantized_one,
        )

        self.pa_input = PhaseAdaptiveInput(NUM_PA_BUCKETS, self.quantized_one)

        self.layer_stacks = LayerStacks(
            NUM_LS_BUCKETS,
            self.quantized_one,
            self.weight_scale_hidden,
            self.weight_scale_out,
            self.score_scale,
        )

        # Weight clipping configuration
        max_hidden_weight = self.quantized_one / self.weight_scale_hidden
        max_out_weight = (self.quantized_one ** 2) / (self.score_scale * self.weight_scale_out)

        self.weight_clipping = [
            {"params": [self.layer_stacks.l1_base.weight], "min_weight": -max_hidden_weight, "max_weight": max_hidden_weight},
            {"params": [self.layer_stacks.l1_pa.weight], "min_weight": -max_hidden_weight, "max_weight": max_hidden_weight},
            {"params": [self.layer_stacks.l2.weight], "min_weight": -max_hidden_weight, "max_weight": max_hidden_weight},
            {"params": [self.layer_stacks.output.weight], "min_weight": -max_out_weight, "max_weight": max_out_weight},
        ]

    def _clip_weights(self) -> None:
        with torch.no_grad():
            for group in self.weight_clipping:
                for param in group["params"]:
                    param.clamp_(group["min_weight"], group["max_weight"])

    def forward(
        self,
        indices: torch.Tensor,
        values: torch.Tensor,
        mobility: torch.Tensor,
        batch_size: int,
        in_features: int,
        ply: torch.Tensor,
    ) -> torch.Tensor:
        # Base input processing
        x_base = self.base_input(indices, values, batch_size, in_features)
        x_base1, x_base2 = torch.split(x_base, LBASE // 2, dim=1)

        # Apply different activations to each half
        x_base1 = fq_floor(x_base1.clamp(0.0, 1.0), 127)
        x_base2 = fq_floor(x_base2.clamp(0.0, 1.0), 127)

        # Combine 
        x_base = fq_floor(x_base1 * x_base2 * 127 / 128, self.quantized_one)

        # Phase-adaptive input
        x_pa = self.pa_input(indices, values, batch_size, in_features, ply)

        # Add mobility features
        mobility_scaled = mobility * 3 / 127
        x_base = torch.cat([x_base, mobility_scaled], dim=1)
        x_pa = torch.cat([x_pa, mobility_scaled], dim=1)

        return self.layer_stacks(x_base, x_pa, ply)

    def _step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
        loss_type: str,
    ) -> torch.Tensor:
        self._clip_weights()

        score_target, feature_indices, mobility, ply = batch
        device = feature_indices.device
        batch_size = feature_indices.size(0)

        # Create sparse representation
        with torch.no_grad():
            batch_indices = torch.arange(batch_size, device=device).repeat_interleave(NUM_FEATURES)
            sparse_indices = torch.stack([batch_indices, feature_indices.view(-1)], dim=0)
            sparse_values = torch.ones(sparse_indices.size(1), device=device)

        score_pred = self(sparse_indices, sparse_values, mobility, batch_size, SUM_OF_FEATURES, ply)
        loss = F.mse_loss(score_pred, score_target)
        self.log(loss_type, loss)
        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._step(batch, batch_idx, "train_loss")

    def validation_step(self, batch, batch_idx: int) -> None:
        self._step(batch, batch_idx, "val_loss")

    def test_step(self, batch, batch_idx: int) -> None:
        self._step(batch, batch_idx, "test_loss")

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LRScheduler]]:
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith(".bias") or param.dim() == 1:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        # Try to use Apex FusedAdam if available, otherwise use PyTorch AdamW
        betas = (0.95, 0.999)
        eps = 1e-4
        try:
            import apex
            optimizer = apex.optimizers.FusedAdam(
                [
                    {"params": decay_params, "weight_decay": self.hparams.weight_decay},
                    {"params": no_decay_params, "weight_decay": 0.0},
                ],
                lr=self.hparams.lr,
                betas=betas,
                eps=eps,
            )
        except ImportError:
            optimizer = torch.optim.AdamW(
                [
                    {"params": decay_params, "weight_decay": self.hparams.weight_decay},
                    {"params": no_decay_params, "weight_decay": 0.0},
                ],
                lr=self.hparams.lr,
                betas=betas,
                eps=eps,
                fused=True,
            )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.t_max,
            eta_min=self.hparams.eta_min,
        )
        return [optimizer], [scheduler]
