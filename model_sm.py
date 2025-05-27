from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from quant import FakeQuantizeLinear, FakeQuantizeSparseLinear, fq_floor


NUM_FEATURE_PARAMS = [
    6561, 6561, 6561, 6561,
    6561, 6561, 6561, 6561,
    6561, 6561, 6561, 6561,
    6561, 6561, 6561, 6561,
    6561, 6561, 6561, 6561,
    6561, 6561,
]


NUM_FEATURES = len(NUM_FEATURE_PARAMS)
SUM_OF_FEATURES = sum(NUM_FEATURE_PARAMS)

LPA = 64
L1_PA = LPA + 1
L2 = 8
L3 = 32
NUM_PA_BUCKETS = 6
NUM_LS_BUCKETS = 60
MAX_PLY = 60


class LayerStacks(nn.Module):
    def __init__(
        self, count, quantized_one, weight_scale_hidden, weight_scale_out, score_scale
    ):
        super(LayerStacks, self).__init__()
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

        self.l1_pa = FakeQuantizeLinear(
            L1_PA,
            L2 * count,
            weight_scale=self.weight_scale_hidden,
            bias_scale=self.weight_scale_hidden * self.quantized_one
        )
        self.l2 = FakeQuantizeLinear(
            L2,
            L3 * count,
            weight_scale=self.weight_scale_hidden,
            bias_scale=self.weight_scale_hidden * self.quantized_one
        )
        self.output = FakeQuantizeLinear(
            L3,
            1 * count,
            weight_scale=self.score_scale * self.weight_scale_out / self.quantized_one,
            bias_scale=self.score_scale * self.weight_scale_out
        )

        # Cached helper tensor for choosing outputs by bucket indices.
        # Initialized lazily in forward.
        self.idx_offset = None

        self._init_layers()

    def _init_layers(self):
        l1_pa_weight = self.l1_pa.weight
        l1_pa_bias = self.l1_pa.bias
        l2_weight = self.l2.weight
        l2_bias = self.l2.bias
        output_weight = self.output.weight
        output_bias = self.output.bias

        with torch.no_grad():
            output_bias.fill_(0.0)

            for i in range(1, self.count):
                # Force all layer stacks to be initialized in the same way.
                l1_pa_weight[i * L2 : (i + 1) * L2, :] = l1_pa_weight[ 0 : L2, :]
                l1_pa_bias[i * L2 : (i + 1) * L2] = l1_pa_bias[0 : L2]
                l2_weight[i * L3 : (i + 1) * L3, :] = l2_weight[0:L3, :]
                l2_bias[i * L3 : (i + 1) * L3] = l2_bias[0:L3]
                output_weight[i : i + 1, :] = output_weight[0:1, :]

        self.l1_pa.weight = nn.Parameter(l1_pa_weight)
        self.l1_pa.bias = nn.Parameter(l1_pa_bias)
        self.l2.weight = nn.Parameter(l2_weight)
        self.l2.bias = nn.Parameter(l2_bias)
        self.output.weight = nn.Parameter(output_weight)
        self.output.bias = nn.Parameter(output_bias)

    def forward(self, x_pa, ply):
        if self.idx_offset is None or self.idx_offset.shape[0] != x_pa.shape[0]:
            self.idx_offset = torch.arange( 0, x_pa.shape[0] * self.count, self.count, device=ply.device )

        ls_indices = ply.flatten() // self.bucket_size
        indices = ls_indices + self.idx_offset

        l1x = self.l1_pa(x_pa).reshape((-1, self.count, L2))
        l1x = l1x.view(-1, L2)[indices]

        l1x = torch.clamp(l1x, 0.0, 1.0)
        l1x = fq_floor(l1x, self.quantized_one)

        l2s = self.l2(l1x).reshape((-1, self.count, L3))
        l2c = l2s.view(-1, L3)[indices]
        l2x = torch.clamp(l2c, 0.0, 1.0)
        l2x = fq_floor(l2x, self.quantized_one)

        l3s = self.output(l2x).reshape((-1, self.count, 1))
        l3x = l3s.view(-1, 1)[indices]
        score_pred = fq_floor(l3x, self.score_scale)

        return score_pred

    def get_coalesced_layer_stacks(self):
        for i in range(self.count):
            with torch.no_grad():
                l1_pa = nn.Linear(L1_PA, L2)
                l2 = nn.Linear(L2 * 2, L3)
                output = nn.Linear(L3, 1)

                l1_pa.weight.data = self.l1_pa.weight[i * L2 : (i + 1) * L2, :]
                l1_pa.bias.data = self.l1_pa.bias[i * L2 : (i + 1) * L2]
                l2.weight.data = self.l2.weight[i * L3 : (i + 1) * L3, :]
                l2.bias.data = self.l2.bias[i * L3 : (i + 1) * L3]
                output.weight.data = self.output.weight[i : (i + 1), :]
                output.bias.data = self.output.bias[i : (i + 1)]

                yield l1_pa, l2, output


class PhaseAdaptiveInput(nn.Module):
    def __init__(self, count, quantized_one):
        super(PhaseAdaptiveInput, self).__init__()
        self.quantized_one = quantized_one
        self.count = count
        self.bucket_size = MAX_PLY // self.count

        self.input = FakeQuantizeSparseLinear(
            SUM_OF_FEATURES,
            LPA * self.count,
            weight_scale=quantized_one,
            bias_scale=quantized_one
        )
        self.idx_offset = None
        self._init_layers()

    def _init_layers(self):
        li_weight = self.input.weight
        li_bias = self.input.bias
        with torch.no_grad():
            for i in range(1, self.count):
                li_weight[:, i * LPA : (i + 1) * LPA] = li_weight[:, 0:LPA]
                li_bias[i * LPA : (i + 1) * LPA] = li_bias[0:LPA]

        self.input.weight = nn.Parameter(li_weight)
        self.input.bias = nn.Parameter(li_bias)

    def forward(self, feature_indices, values, m, n, ply):
        if self.idx_offset is None or self.idx_offset.shape[0] != m:
            self.idx_offset = torch.arange(
                0, m * self.count, self.count, device=ply.device
            )
        x = self.input(feature_indices, values, m, n).reshape((-1, self.count, LPA))

        bucket_idx = ply.flatten() // self.bucket_size + self.idx_offset
        x = x.view(-1, LPA)[bucket_idx]

        x = F.leaky_relu(x, negative_slope=0.125)
        x = torch.clamp(x, -16 / 127, 1.0 - 16 / 127)
        x = torch.add(x, 16 / 127)
        x = fq_floor(x, self.quantized_one, 0.0, 1.0)
        return x

    def get_layers(self):
        for i in range(self.count):
            with torch.no_grad():
                li = nn.Linear(SUM_OF_FEATURES, LPA)
                li.weight.data = self.input.weight[:, i * LPA : (i + 1) * LPA]
                li.bias.data = self.input.bias[i * LPA : (i + 1) * LPA]
                yield li


class ReversiSmallModel(L.LightningModule):
    def __init__(
        self,
        lr: float = 0.001,
        t_max: int = 100,
        eta_min: float = 1e-9,
    ):
        super(ReversiSmallModel, self).__init__()
        self.save_hyperparameters()

        # Quantisation constants
        self.score_scale = 128.0
        self.weight_scale_hidden = 64.0
        self.weight_scale_out = 16.0
        self.quantized_one = 127.0

        self.pa_input = PhaseAdaptiveInput(NUM_PA_BUCKETS, self.quantized_one)
        self.layer_stacks = LayerStacks(
            NUM_LS_BUCKETS,
            self.quantized_one,
            self.weight_scale_hidden,
            self.weight_scale_out,
            self.score_scale,
        )

        # compile
        self.pa_input = torch.compile(self.pa_input)
        self.layer_stacks = torch.compile(self.layer_stacks)

        max_hidden_weight = self.quantized_one / self.weight_scale_hidden
        max_out_weight = (self.quantized_one * self.quantized_one) / (
            self.score_scale * self.weight_scale_out
        )
        self.weight_clipping = [
            {
                "params": [self.layer_stacks.l1_pa.weight],
                "min_weight": -max_hidden_weight,
                "max_weight": max_hidden_weight,
            },
            {
                "params": [self.layer_stacks.l2.weight],
                "min_weight": -max_hidden_weight,
                "max_weight": max_hidden_weight,
            },
            {
                "params": [self.layer_stacks.output.weight],
                "min_weight": -max_out_weight,
                "max_weight": max_out_weight,
            },
        ]

    def _clip_weights(self):
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
        x_pa = self.pa_input(indices, values, batch_size, in_features, ply)
        x_pa = torch.cat([x_pa, mobility * 3 / 127], dim=1)

        return self.layer_stacks(x_pa, ply)

    def _step(self, batch, batch_idx, loss_type):
        self._clip_weights()

        score_target, feature_indices, mobility, ply = batch
        device = feature_indices.device
        batch_size = feature_indices.size(0)

        batch_indices = torch.arange(batch_size, device=device).repeat_interleave(NUM_FEATURES)
        sparse_indices = torch.stack([batch_indices, feature_indices.view(-1)], dim=0)
        sparse_values = torch.ones(sparse_indices.size(1), device=device)

        score_pred = self(
            sparse_indices, sparse_values, mobility, batch_size, SUM_OF_FEATURES, ply
        )

        loss = F.mse_loss(score_pred, score_target)
        self.log(loss_type, loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train_loss")

    def validation_step(self, batch, batch_idx):
        self._step(batch, batch_idx, "val_loss")

    def test_step(self, batch, batch_idx):
        self._step(batch, batch_idx, "test_loss")

    def configure_optimizers(
        self,
    ) -> Tuple[
        List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]
    ]:
        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if name.endswith(".bias") or param.dim() == 1:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": 1e-2},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.hparams.lr,
            betas=(0.95, 0.999),
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.t_max,
            eta_min=self.hparams.eta_min,
        )
        return [optimizer], [scheduler]
