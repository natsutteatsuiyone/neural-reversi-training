import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from typing import Tuple, List

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

        self.idx_offset = None

        self._init_layers()

    def _init_layers(self) -> None:
        """
        レイヤーの初期化を行う。
        すべてのレイヤースタックが同じ方法で初期化されるようにする。
        """
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
                # Force all layer stacks to be initialized in the same way.
                l1_base_weight[i * L2_HALF : (i + 1) * L2_HALF, :] = l1_base_weight[0:L2_HALF, :]
                l1_base_bias[i * L2_HALF : (i + 1) * L2_HALF] = l1_base_bias[0:L2_HALF]
                l1_pa_weight[i * L2_HALF : (i + 1) * L2_HALF, :] = l1_pa_weight[0:L2_HALF, :]
                l1_pa_bias[i * L2_HALF : (i + 1) * L2_HALF] = l1_pa_bias[0:L2_HALF]
                l2_weight[i * L3 : (i + 1) * L3, :] = l2_weight[0:L3, :]
                l2_bias[i * L3 : (i + 1) * L3] = l2_bias[0:L3]
                output_weight[i : i + 1, :] = output_weight[0:1, :]
                output_bias[i : i + 1] = output_bias[0:1]

        self.l1_base.weight = nn.Parameter(l1_base_weight)
        self.l1_base.bias = nn.Parameter(l1_base_bias)
        self.l1_pa.weight = nn.Parameter(l1_pa_weight)
        self.l1_pa.bias = nn.Parameter(l1_pa_bias)
        self.l2.weight = nn.Parameter(l2_weight)
        self.l2.bias = nn.Parameter(l2_bias)
        self.output.weight = nn.Parameter(output_weight)
        self.output.bias = nn.Parameter(output_bias)

    def forward(self, x_base: torch.Tensor, x_pa: torch.Tensor, ply: torch.Tensor) -> torch.Tensor:
        if self.idx_offset is None or self.idx_offset.shape[0] != x_base.shape[0]:
            self.idx_offset = torch.arange(
                0, x_base.shape[0] * self.count, self.count, device=ply.device
            )

        ls_indices = ply.flatten() // self.bucket_size
        indices = ls_indices + self.idx_offset

        l1x_base = self.l1_base(x_base).reshape((-1, self.count, L2 // 2))
        l1x_base = l1x_base.view(-1, L2 // 2)[indices]

        l1x_pa = self.l1_pa(x_pa).reshape((-1, self.count, L2 // 2))
        l1x_pa = l1x_pa.view(-1, L2 // 2)[indices]

        l1x = torch.cat([l1x_base, l1x_pa], dim=1)

        l1x = torch.clamp(
            torch.cat([torch.pow(l1x, 2.0) * (127 / 128), l1x], dim=1), 0.0, 1.0
        )
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
                l1_base = nn.Linear(L1_BASE, L2_HALF)
                l1_pa = nn.Linear(L1_PA, L2_HALF)
                l2 = nn.Linear(L2 * 2, L3)
                output = nn.Linear(L3, 1)

                l1_base.weight.data = self.l1_base.weight[i * L2_HALF : (i + 1) * L2_HALF, :]
                l1_base.bias.data = self.l1_base.bias[i * L2_HALF : (i + 1) * L2_HALF]
                l1_pa.weight.data = self.l1_pa.weight[i * L2_HALF : (i + 1) * L2_HALF, :]
                l1_pa.bias.data = self.l1_pa.bias[i * L2_HALF : (i + 1) * L2_HALF]
                l2.weight.data = self.l2.weight[i * L3 : (i + 1) * L3, :]
                l2.bias.data = self.l2.bias[i * L3 : (i + 1) * L3]
                output.weight.data = self.output.weight[i : (i + 1), :]
                output.bias.data = self.output.bias[i : (i + 1)]

                yield l1_base, l1_pa, l2, output


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
            bias_scale=quantized_one,
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


class ReversiModel(L.LightningModule):
    def __init__(
        self,
        lr: float = 0.001,
        t_max: int = 100,
        eta_min: float = 1e-9,
    ):
        super(ReversiModel, self).__init__()
        self.save_hyperparameters()

        # Quantisation constants
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

        # compile
        self.base_input = torch.compile(self.base_input)
        self.pa_input = torch.compile(self.pa_input)
        self.layer_stacks = torch.compile(self.layer_stacks)

        max_hidden_weight = self.quantized_one / self.weight_scale_hidden
        max_out_weight = (self.quantized_one * self.quantized_one) / (
            self.score_scale * self.weight_scale_out
        )
        self.weight_clipping = [
            {
                "params": [self.layer_stacks.l1_base.weight],
                "min_weight": -max_hidden_weight,
                "max_weight": max_hidden_weight,
            },
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
        x_base = self.base_input(indices, values, batch_size, in_features)
        x_base_s = torch.split(x_base, LBASE // 2, dim=1)

        # Clipped ReLU
        x_base_s0 = torch.clamp(x_base_s[0], 0.0, 1.0)
        x_base_s0 = fq_floor(x_base_s0, 127)

        # Hard sigmoid
        x_base_s1 = torch.clamp(x_base_s[1] * 0.25 + 0.5, 0.0, 1.0)
        x_base_s1 = fq_floor(x_base_s1, self.quantized_one * 2)

        x_base = x_base_s0 * x_base_s1 * 127 / 128
        x_base = fq_floor(x_base, self.quantized_one)
        x_base = torch.cat([x_base, mobility * 3 / 127], dim=1)

        x_pa = self.pa_input(indices, values, batch_size, in_features, ply)
        x_pa = torch.cat([x_pa, mobility * 3 / 127], dim=1)

        return self.layer_stacks(x_base, x_pa, ply)

    def _step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
        loss_type: str,
    ) -> torch.Tensor:
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
        self.log(f"{loss_type}", loss)
        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._step(batch, batch_idx, "train_loss")

    def validation_step(self, batch, batch_idx: int) -> None:
        self._step(batch, batch_idx, "val_loss")

    def test_step(self, batch, batch_idx: int) -> None:
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
