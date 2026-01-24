import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from typing import Tuple, List, Iterator

from model_common import (
    SUM_OF_FEATURES,
    bucket_lookup_indices,
    repeat_first_block,
    select_bucket,
)
from sparse_linear import SparseLinear

LINPUT = 256

NUM_LS_BUCKETS = 60
MAX_PLY = 60

class LayerStacks(nn.Module):
    def __init__(
        self,
        count: int,
    ):
        super().__init__()
        self.count = count
        self.bucket_size = MAX_PLY // count

        self.output = nn.Linear(
            LINPUT,
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
                output = nn.Linear(LINPUT, 1)

                output.weight.data = self.output.weight[i:i+1]
                output.bias.data = self.output.bias[i:i+1]

                yield output

class ReversiWasmlModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # Quantization constants
        self.score_scale = 64.0
        self.eval_score_scale = 256.0
        self.weight_scale_out = self.eval_score_scale * 128.0
        self.quantized_one = 255.0

        self.max_input_weight = 127.0 / self.quantized_one

        self.input = SparseLinear(
            SUM_OF_FEATURES,
            LINPUT
        )
        self.layer_stacks = LayerStacks(NUM_LS_BUCKETS)

    def forward(
        self,
        feature_indices: torch.Tensor,
        mobility: torch.Tensor,
        ply: torch.Tensor,
    ) -> torch.Tensor:
        x = self.input(feature_indices)
        x = x.clamp(0.0, 1.0).pow(2.0) * (255.0 / 256.0)
        return self.layer_stacks(x, ply)


class LitReversiWasmModel(L.LightningModule):
    def __init__(
        self,
        lr: float = 0.001,
        weight_decay: float = 1e-2,
        t_max: int = 100,
        eta_min: float = 1e-9,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = ReversiWasmlModel()
        self.weight_clipping = [
            {
                "params": [self.model.input.weight],
                "min_weight": -self.model.max_input_weight,
                "max_weight": self.model.max_input_weight,
            },
        ]

    def forward(
        self,
        feature_indices: torch.Tensor,
        mobility: torch.Tensor,
        ply: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(feature_indices, mobility, ply)

    @torch.compile(fullgraph=True, options={"shape_padding": True})
    def _step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        score_target, feature_indices, mobility, ply = batch

        score_pred = self(feature_indices, mobility, ply)
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

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        optimizer.step(closure=optimizer_closure)

        # clip weights
        with torch.no_grad():
            for g in self.weight_clipping:
                for p in g["params"]:
                    p.clamp_(g["min_weight"], g["max_weight"])

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
