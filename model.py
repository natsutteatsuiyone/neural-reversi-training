import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from dataset import NUM_FEATURES, SUM_OF_FEATURES
from quant import q_floor, q_round
from space_linear import SparseLinear, sparse_linear

LBASE = 512
L1_BASE = (LBASE // 2) + 1
L1_PS = 128
L2 = 16
L3 = 64
NUM_PS_BUCKETS = 6
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

        self.l1_base = nn.Linear(L1_BASE, (L2 // 2) * count)
        self.l1_ps = nn.Linear(L1_PS, (L2 // 2) * count)
        self.l2 = nn.Linear(L2 * 2, L3 * count)
        self.output = nn.Linear(L3, 1 * count)

        # Cached helper tensor for choosing outputs by bucket indices.
        # Initialized lazily in forward.
        self.idx_offset = None

        self._init_layers()

    def _init_layers(self):
        l1_base_weight = self.l1_base.weight
        l1_base_bias = self.l1_base.bias
        l1_ps_weight = self.l1_ps.weight
        l1_ps_bias = self.l1_ps.bias
        l2_weight = self.l2.weight
        l2_bias = self.l2.bias
        output_weight = self.output.weight
        output_bias = self.output.bias
        with torch.no_grad():
            output_bias.fill_(0.0)

            for i in range(1, self.count):
                # Force all layer stacks to be initialized in the same way.
                l1_base_weight[i * (L2 // 2) : (i + 1) * (L2 // 2), :] = l1_base_weight[ 0 : (L2 // 2), :]
                l1_base_bias[i * (L2 // 2) : (i + 1) * (L2 // 2)] = l1_base_bias[0 : (L2 // 2)]
                l1_ps_weight[i * (L2 // 2) : (i + 1) * (L2 // 2), :] = l1_ps_weight[ 0 : (L2 // 2), :]
                l1_ps_bias[i * (L2 // 2) : (i + 1) * (L2 // 2)] = l1_ps_bias[0 : (L2 // 2)]
                l2_weight[i * L3 : (i + 1) * L3, :] = l2_weight[0:L3, :]
                l2_bias[i * L3 : (i + 1) * L3] = l2_bias[0:L3]
                output_weight[i : i + 1, :] = output_weight[0:1, :]

        self.l1_base.weight = nn.Parameter(l1_base_weight)
        self.l1_base.bias = nn.Parameter(l1_base_bias)
        self.l1_ps.weight = nn.Parameter(l1_ps_weight)
        self.l1_ps.bias = nn.Parameter(l1_ps_bias)
        self.l2.weight = nn.Parameter(l2_weight)
        self.l2.bias = nn.Parameter(l2_bias)
        self.output.weight = nn.Parameter(output_weight)
        self.output.bias = nn.Parameter(output_bias)

    def forward(self, x_base, x_ps, ply):
        if self.idx_offset is None or self.idx_offset.shape[0] != x_base.shape[0]:
            self.idx_offset = torch.arange( 0, x_base.shape[0] * self.count, self.count, device=ply.device )

        ls_indices = ply.flatten() // self.bucket_size
        indices = ls_indices + self.idx_offset

        q_l1_weight = q_round(self.l1_base.weight, self.weight_scale_hidden)
        q_l1_bias = q_round(self.l1_base.bias, self.weight_scale_hidden * self.quantized_one)
        l1x_base = F.linear(x_base, q_l1_weight, q_l1_bias).reshape((-1, self.count, L2 // 2))
        l1x_base = l1x_base.view(-1, L2 // 2)[indices]

        q_li_ps_weight = q_round(self.l1_ps.weight, self.weight_scale_hidden)
        q_li_ps_bias = q_round(self.l1_ps.bias, self.weight_scale_hidden * self.quantized_one)
        l1x_ps = F.linear(x_ps, q_li_ps_weight, q_li_ps_bias).reshape((-1, self.count, L2 // 2))
        l1x_ps = l1x_ps.view(-1, L2 // 2)[indices]

        l1x = torch.cat([l1x_base, l1x_ps], dim=1)

        # multiply sqr crelu result by (127/128) to match quantized version
        l1x = torch.clamp( torch.cat([torch.pow(l1x, 2.0) * (127 / 128), l1x], dim=1), 0.0, 1.0)
        l1x = q_floor(l1x, self.quantized_one)

        q_l2_weight = q_round(self.l2.weight, self.weight_scale_hidden)
        q_l2_bias = q_round(self.l2.bias, self.weight_scale_hidden * self.quantized_one)
        l2s_ = F.linear(l1x, q_l2_weight, q_l2_bias).reshape((-1, self.count, L3))
        l2c_ = l2s_.view(-1, L3)[indices]

        l2x_ = torch.clamp(l2c_, 0.0, 1.0)
        l2x_ = q_floor(l2x_, self.quantized_one)

        q_output_weight = q_round(
            self.output.weight,
            self.score_scale * self.weight_scale_out / self.quantized_one,
        )
        q_output_bias = q_round(
            self.output.bias, self.score_scale * self.weight_scale_out
        )
        l3s_ = F.linear(l2x_, q_output_weight, q_output_bias).reshape(
            (-1, self.count, 1)
        )
        l3c_ = l3s_.view(-1, 1)[indices]
        l3x_ = l3c_

        output = q_floor(l3x_, self.score_scale)
        return output

    def get_coalesced_layer_stacks(self):
        for i in range(self.count):
            with torch.no_grad():
                l1_base = nn.Linear(L1_BASE, (L2 // 2))
                l1_ps = nn.Linear(L1_PS, (L2 // 2))
                l2 = nn.Linear(L2 * 2, L3)
                output = nn.Linear(L3, 1)
                l1_base.weight.data = self.l1_base.weight[i * (L2 // 2) : (i + 1) * (L2 // 2), :]
                l1_base.bias.data = self.l1_base.bias[i * (L2 // 2) : (i + 1) * (L2 // 2)]
                l1_ps.weight.data = self.l1_ps.weight[i * (L2 // 2) : (i + 1) * (L2 // 2), :]
                l1_ps.bias.data = self.l1_ps.bias[i * (L2 // 2) : (i + 1) * (L2 // 2)]
                l2.weight.data = self.l2.weight[i * L3 : (i + 1) * L3, :]
                l2.bias.data = self.l2.bias[i * L3 : (i + 1) * L3]
                output.weight.data = self.output.weight[i : (i + 1), :]
                output.bias.data = self.output.bias[i : (i + 1)]
                yield l1_base, l1_ps, l2, output


class PhaseSpecificInput(nn.Module):
    def __init__(self, count, quantized_one):
        super(PhaseSpecificInput, self).__init__()
        self.quantized_one = quantized_one
        self.count = count
        self.bucket_size = MAX_PLY // self.count
        self.input = SparseLinear(SUM_OF_FEATURES, L1_PS * self.count)
        self.idx_offset = None
        self._init_layers()

    def _init_layers(self):
        li_weight = self.input.weight
        li_bias = self.input.bias
        with torch.no_grad():
            for i in range(1, self.count):
                li_weight[:, i * L1_PS : (i + 1) * L1_PS] = li_weight[:, 0:L1_PS]
                li_bias[i * L1_PS : (i + 1) * L1_PS] = li_bias[0:L1_PS]

        self.input.weight = nn.Parameter(li_weight)
        self.input.bias = nn.Parameter(li_bias)

    def forward(self, feature_indices, values, m, n, ply):
        if self.idx_offset is None or self.idx_offset.shape[0] != m:
            self.idx_offset = torch.arange(
                0, m * self.count, self.count, device=ply.device
            )
        q_weight = q_round(self.input.weight, self.quantized_one)
        q_bias = q_round(self.input.bias, self.quantized_one)
        x = sparse_linear(feature_indices, values, m, n, q_weight, q_bias).reshape(
            (-1, self.count, L1_PS)
        )

        indicies = ply.flatten() // self.bucket_size + self.idx_offset
        x = x.view(-1, L1_PS)[indicies]
        x = torch.clamp(x, 0.0, 1.0)
        return x

    def get_layers(self):
        for i in range(self.count):
            with torch.no_grad():
                li = nn.Linear(SUM_OF_FEATURES, L1_PS)
                li.weight.data = self.input.weight[:, i * L1_PS : (i + 1) * L1_PS]
                li.bias.data = self.input.bias[i * L1_PS : (i + 1) * L1_PS]
                yield li


class ReversiModel(L.LightningModule):
    def __init__(
        self,
        lr=0.001,
        t_max=100,
        eta_min=1e-10,
    ):
        super(ReversiModel, self).__init__()

        self.lr = lr
        self.t_max = t_max
        self.eta_min = eta_min

        self.score_scale = 64.0
        self.weight_scale_hidden = 64.0
        self.weight_scale_out = 16.0
        self.quantized_one = 127.0

        self.input = SparseLinear(SUM_OF_FEATURES, LBASE)
        self.ps_input = PhaseSpecificInput(NUM_PS_BUCKETS, self.quantized_one)
        self.layer_stacks = LayerStacks(
            NUM_LS_BUCKETS,
            self.quantized_one,
            self.weight_scale_hidden,
            self.weight_scale_out,
            self.score_scale,
        )

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

    def forward(self, feature_indices, values, m, n, mobility, ply):
        q_input_weight = q_round(self.input.weight, self.quantized_one)
        q_input_bias = q_round(self.input.bias, self.quantized_one)
        x_base = sparse_linear(feature_indices, values, m, n, q_input_weight, q_input_bias)
        x_base = torch.clamp(x_base, 0.0, 1.0)

        x_base_s = torch.split(x_base, LBASE // 2, dim=1)
        x_base = x_base_s[0] * x_base_s[1] * 127 / 128
        x_base = q_floor(x_base, self.quantized_one)
        x_base = torch.cat([x_base, mobility * 3 / 127], dim=1)

        x_ps = self.ps_input(feature_indices, values, m, n, ply)

        return self.layer_stacks(x_base, x_ps, ply)

    def step_(self, batch, batch_idx, loss_type):
        self._clip_weights()

        feature_indices, mobility, scores, ply = batch
        device = feature_indices.device
        batch_size = feature_indices.size(0)

        batch_indices = torch.arange(batch_size, device=device).repeat_interleave(
            NUM_FEATURES
        )
        flat_feature_indices = feature_indices.view(-1)
        sparse_indices = torch.stack([batch_indices, flat_feature_indices], dim=0)
        sparse_values = torch.ones(sparse_indices.size(1), device=device)

        outputs = self(
            sparse_indices,
            sparse_values,
            batch_size,
            SUM_OF_FEATURES,
            mobility.unsqueeze(-1),
            ply.unsqueeze(-1),
        )

        loss = F.mse_loss(outputs, scores.unsqueeze(-1))
        self.log(loss_type, loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step_(batch, batch_idx, "train_loss")

    def validation_step(self, batch, batch_idx):
        self.step_(batch, batch_idx, "val_loss")

    def test_step(self, batch, batch_idx):
        self.step_(batch, batch_idx, "test_loss")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, betas=(0.95, 0.999)
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.t_max,
            eta_min=self.eta_min,
        )
        return [optimizer], [scheduler]
