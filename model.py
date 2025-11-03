from typing import Any, Dict, Iterator, List, Tuple

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_common import (
    NUM_FEATURES,
    SUM_OF_FEATURES,
    bucket_lookup_indices,
    repeat_first_block,
    select_bucket,
)
from sparse_linear import SparseLinear

LBASE = 256
LPA = 128
L1_BASE = (LBASE // 2) + 1
L1_PA = LPA + 1
L2 = 16
L2_HALF = L2 // 2
L3 = 64
LB = 128
LOUTPUT = L3 + LPA + (LBASE // 2)

NUM_PA_BUCKETS = 6
NUM_LS_BUCKETS = 60
MAX_PLY = 60


ParamGroup = Dict[str, Any]


class LayerStacks(nn.Module):
    def __init__(
        self,
        count: int,
    ):
        super().__init__()
        self.count = count
        self.bucket_size = MAX_PLY // count

        self.l1_base = nn.Linear(
            L1_BASE,
            L2_HALF * count,
        )
        self.l1_pa = nn.Linear(
            L1_PA,
            L2_HALF * count,
        )
        self.l2 = nn.Linear(
            L2 * 2,
            L3 * count,
        )
        self.output = nn.Linear(
            LOUTPUT,
            1 * count,
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
            output_bias.zero_()

            repeat_first_block(l1_base_weight, L2_HALF, self.count)
            repeat_first_block(l1_base_bias, L2_HALF, self.count)
            repeat_first_block(l1_pa_weight, L2_HALF, self.count)
            repeat_first_block(l1_pa_bias, L2_HALF, self.count)
            repeat_first_block(l2_weight, L3, self.count)
            repeat_first_block(l2_bias, L3, self.count)
            repeat_first_block(output_weight, 1, self.count)
            repeat_first_block(output_bias, 1, self.count)

    def forward(
        self,
        x_base: torch.Tensor,
        x_pa: torch.Tensor,
        mobility: torch.Tensor,
        ply: torch.Tensor,
    ) -> torch.Tensor:
        bucket_indices = bucket_lookup_indices(ply, self.bucket_size, self.count)

        # Add mobility features
        mobility_scaled = torch.clamp(mobility * (7.0 / 255.0), max=1.0)
        x_base_with_mobility = torch.cat([x_base, mobility_scaled], dim=1)
        x_pa_with_mobility = torch.cat([x_pa, mobility_scaled], dim=1)

        # Process base features
        l1x_base = self.l1_base(x_base_with_mobility)
        l1x_base = select_bucket(l1x_base, L2_HALF, bucket_indices)

        # Process PA features
        l1x_pa = self.l1_pa(x_pa_with_mobility)
        l1x_pa = select_bucket(l1x_pa, L2_HALF, bucket_indices)

        # Combine and apply activations
        l1x = torch.cat([l1x_base, l1x_pa], dim=1)
        l1x_squared = l1x.pow(2) * (255 / 256)
        l1x = torch.cat([l1x_squared, l1x], dim=1)
        l1x = torch.clamp(l1x, 0.0, 1.0)

        # Second layer
        l2x = self.l2(l1x)
        l2x = select_bucket(l2x, L3, bucket_indices)
        l2x = torch.clamp(l2x, 0.0, 1.0).pow(2) * (255 / 256)

        # Output layer
        output_features = torch.cat([l2x, x_base, x_pa], dim=1)
        output = self.output(output_features)
        output = select_bucket(output, 1, bucket_indices)

        return output

    def get_layer_stacks(self) -> Iterator[Tuple[nn.Linear, nn.Linear, nn.Linear, nn.Linear]]:
        """Extract individual layer stacks as separate nn.Linear modules."""
        for i in range(self.count):
            with torch.no_grad():
                l1_base = nn.Linear(L1_BASE, L2_HALF)
                l1_pa = nn.Linear(L1_PA, L2_HALF)
                l2 = nn.Linear(L2 * 2, L3)
                output = nn.Linear(LOUTPUT, 1)

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
    def __init__(self, count: int):
        super().__init__()
        self.count = count
        self.bucket_size = MAX_PLY // count

        self.input = SparseLinear(
            SUM_OF_FEATURES,
            LPA * self.count,
        )
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize layers to ensure all phase-adaptive buckets are initialized identically."""
        with torch.no_grad():
            repeat_first_block(self.input.weight, LPA, self.count, dim=1)
            repeat_first_block(self.input.bias, LPA, self.count)

    def forward(
        self,
        feature_indices: torch.Tensor,
        values: torch.Tensor,
        batch_size: int,
        in_features: int,
        ply: torch.Tensor
    ) -> torch.Tensor:
        x = self.input(feature_indices, values, batch_size, in_features)
        bucket_indices = bucket_lookup_indices(ply, self.bucket_size, self.count)
        x = select_bucket(x, LPA, bucket_indices)
        return x.clamp(0.0, 1.0).pow(2.0) * (255 / 256)

    def get_layers(self) -> Iterator[nn.Linear]:
        """Extract individual phase-adaptive layers as separate nn.Linear modules."""
        for i in range(self.count):
            with torch.no_grad():
                layer = nn.Linear(SUM_OF_FEATURES, LPA)
                start, end = i * LPA, (i + 1) * LPA
                layer.weight.data = self.input.weight[:, start:end]
                layer.bias.data = self.input.bias[start:end]
                yield layer


class ReversiModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # Quantization constants
        self.score_scale = 64.0
        self.eval_score_scale = 256.0
        self.weight_scale_hidden = 64.0
        self.weight_scale_out = self.eval_score_scale * 64.0
        self.quantized_one = 255.0
        self.quantized_weight_max = 127.0

        # Weight clipping limit
        self.max_hidden_weight = self.quantized_weight_max / self.weight_scale_hidden

        # Networks
        self.base_input = SparseLinear(SUM_OF_FEATURES, LBASE)
        self.pa_input = PhaseAdaptiveInput(NUM_PA_BUCKETS)
        self.layer_stacks = LayerStacks(NUM_LS_BUCKETS)

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
        x_base1 = torch.clamp(x_base1, 0.0, 1.0)
        x_base2 = torch.clamp(x_base2, 0.0, 1.0)

        # Combine
        x_base = x_base1 * x_base2 * (255 / 256)

        # Phase-adaptive input
        x_pa = self.pa_input(indices, values, batch_size, in_features, ply)

        return self.layer_stacks(x_base, x_pa, mobility, ply)

class LitReversiModel(L.LightningModule):
    def __init__(
        self,
        lr: float = 0.001,
        weight_decay: float = 1e-2,
        t_max: int = 100,
        eta_min: float = 1e-8,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Core model
        self.model = ReversiModel()

        self.weight_clipping = [
            {
                "params": [self.model.layer_stacks.l1_base.weight],
                "min_weight": -self.model.max_hidden_weight,
                "max_weight": self.model.max_hidden_weight,
            },
            {
                "params": [self.model.layer_stacks.l1_pa.weight],
                "min_weight": -self.model.max_hidden_weight,
                "max_weight": self.model.max_hidden_weight,
            },
            {
                "params": [self.model.layer_stacks.l2.weight],
                "min_weight": -self.model.max_hidden_weight,
                "max_weight": self.model.max_hidden_weight,
            }
        ]

    def forward(
        self,
        indices: torch.Tensor,
        values: torch.Tensor,
        mobility: torch.Tensor,
        batch_size: int,
        in_features: int,
        ply: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(indices, values, mobility, batch_size, in_features, ply)

    @torch.compile(fullgraph=True, options={"shape_padding": True, "triton.cudagraphs": True})
    def _step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        score_target, feature_indices, mobility, ply = batch
        device = feature_indices.device
        batch_size = feature_indices.size(0)

        # Create sparse representation
        with torch.no_grad():
            batch_indices = torch.arange(batch_size, device=device).repeat_interleave(NUM_FEATURES)
            sparse_indices = torch.stack([batch_indices, feature_indices.view(-1)], dim=0)
            sparse_values = torch.ones(sparse_indices.size(1), device=device)

        score_pred = self(sparse_indices, sparse_values, mobility, batch_size, SUM_OF_FEATURES, ply)
        score_target_scaled = score_target / self.model.score_scale
        return F.mse_loss(score_pred, score_target_scaled)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss = self._step(batch, batch_idx)
        return { "loss": loss }

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.log_dict({
            "train_loss": outputs["loss"],
        })

    def validation_step(self, batch, batch_idx: int) -> None:
        loss = self._step(batch, batch_idx)
        self.log("val_loss", loss)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        optimizer.step(closure=optimizer_closure)

        # clip weights
        with torch.no_grad():
            for g in self.weight_clipping:
                for p in g["params"]:
                    p.clamp_(g["min_weight"], g["max_weight"])

    def _build_param_groups(self) -> List[ParamGroup]:
        sparse_weight_suffixes = {"base_input.weight", "pa_input.input.weight"}
        param_buckets: Dict[str, List[nn.Parameter]] = {
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

    def configure_optimizers(self) -> Dict[str, Any]:
        param_groups = self._build_param_groups()

        betas = (0.9, 0.999)
        eps = 1e-8

        try:
            import apex.optimizers

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

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
