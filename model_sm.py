import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from typing import Tuple, List, Iterator

from model_common import (
    NUM_FEATURES,
    SUM_OF_FEATURES,
    bucket_lookup_indices,
    repeat_first_block,
    select_bucket,
)
from sparse_linear import SparseLinear

LPA = 128
LOUTPUT = LPA

NUM_PA_BUCKETS = 3
NUM_LS_BUCKETS = 30
MAX_PLY = 30

class LayerStacks(nn.Module):
    def __init__(
        self,
        count: int,
    ):
        super().__init__()
        self.count = count
        self.bucket_size = MAX_PLY // count

        self.output = nn.Linear(
            LOUTPUT,
            1 * count,
        )

        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize layers to ensure all layer stacks are initialized identically."""
        output_weight = self.output.weight
        output_bias = self.output.bias

        with torch.no_grad():
            output_bias.zero_()

            repeat_first_block(output_weight, 1, self.count)
            repeat_first_block(output_bias, 1, self.count)

    def forward(self, x_pa: torch.Tensor, ply: torch.Tensor) -> torch.Tensor:
        bucket_indices = bucket_lookup_indices(ply, self.bucket_size, self.count)
        output = self.output(x_pa)
        output = select_bucket(output, 1, bucket_indices)
        return output

    def get_layer_stacks(self) -> Iterator[Tuple[nn.Linear]]:
        """Extract individual layer stacks as separate nn.Linear modules."""
        for i in range(self.count):
            with torch.no_grad():
                output = nn.Linear(LOUTPUT, 1)

                output.weight.data = self.output.weight[i:i+1]
                output.bias.data = self.output.bias[i:i+1]

                yield output


class PhaseAdaptiveInput(nn.Module):
    def __init__(self, count: int):
        super().__init__()
        self.count = count
        self.bucket_size = MAX_PLY // self.count

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
        x = x.clamp(0.0, 1.0).pow(2.0) * (1023 / 1024)
        return x

    def get_layers(self) -> Iterator[nn.Linear]:
        """Extract individual phase-adaptive layers as separate nn.Linear modules."""
        for i in range(self.count):
            with torch.no_grad():
                layer = nn.Linear(SUM_OF_FEATURES, LPA)
                start, end = i * LPA, (i + 1) * LPA
                layer.weight.data = self.input.weight[:, start:end]
                layer.bias.data = self.input.bias[start:end]
                yield layer


class ReversiSmallModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # Quantization constants
        self.score_scale = 64.0
        self.eval_score_scale = 256.0
        self.weight_scale_hidden = 64.0
        self.weight_scale_out = self.eval_score_scale * 256.0
        self.quantized_one = 1023.0

        self.pa_input = PhaseAdaptiveInput(NUM_PA_BUCKETS)
        self.layer_stacks = LayerStacks(NUM_LS_BUCKETS)

    def forward(
        self,
        indices: torch.Tensor,
        values: torch.Tensor,
        batch_size: int,
        in_features: int,
        ply: torch.Tensor,
    ) -> torch.Tensor:
        x_pa = self.pa_input(indices, values, batch_size, in_features, ply)
        return self.layer_stacks(x_pa, ply)


class LitReversiSmallModel(L.LightningModule):
    def __init__(
        self,
        lr: float = 0.001,
        weight_decay: float = 1e-2,
        t_max: int = 100,
        eta_min: float = 1e-9,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Core model
        self.model = ReversiSmallModel()

    def forward(
        self,
        indices: torch.Tensor,
        values: torch.Tensor,
        batch_size: int,
        in_features: int,
        ply: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(indices, values, batch_size, in_features, ply)

    @torch.compile(fullgraph=True, options={"shape_padding": True, "triton.cudagraphs": True})
    def _step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        score_target, feature_indices, _mobility, ply = batch
        device = feature_indices.device
        batch_size = feature_indices.size(0)
        ply = ply.sub(30)

        with torch.no_grad():
            batch_indices = torch.arange(batch_size, device=device).repeat_interleave(NUM_FEATURES)
            sparse_indices = torch.stack([batch_indices, feature_indices.view(-1)], dim=0)
            sparse_values = torch.ones(sparse_indices.size(1), device=device)

        score_pred = self(sparse_indices, sparse_values, batch_size, SUM_OF_FEATURES, ply)
        return F.mse_loss(score_pred, score_target / self.model.score_scale)

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

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LRScheduler]]:
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
            optimizer = apex.optimizers.FusedAdam(
                params,
                lr=self.hparams.lr,
                betas=betas,
                eps=eps,
            )
        except ImportError:
            optimizer = torch.optim.AdamW(
                params,
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
