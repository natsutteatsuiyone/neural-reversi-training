"""Large model (ReversiModel) serialization writer."""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import model_lg
from models.serialization.serialize_common import (
    PADDING_ALIGNMENT,
    BaseNNWriter,
    quantize_tensor,
)


class LargeNNWriter(BaseNNWriter):
    """Serialize ``ReversiModel`` instances into the binary NNUE format."""

    def __init__(self, model: model_lg.ReversiModel, show_hist: bool = True) -> None:
        super().__init__(show_hist)
        self.write_model(model)

    def write_model(self, model: Any) -> None:
        """Write the large model to the internal buffer."""
        self.write_input_layer(model)
        for l1_base, l1_ps, l2, output in model.layer_stacks.get_layer_stacks():
            self.write_fc_layer(model, l1_base)
            self.write_fc_layer(model, l1_ps)
            self.write_fc_layer(model, l2)
            self.write_fc_layer(model, output, is_output=True)

    def write_input_layer(self, model: model_lg.ReversiModel) -> None:
        """Write the input layer to the buffer."""
        base_layer = model.base_input
        scale = model.quantized_one * 2
        ft_bias = quantize_tensor(base_layer.bias, scale, torch.int16)
        ft_weight = quantize_tensor(base_layer.weight, scale, torch.int16)
        self._write_dense_block("ft", ft_bias, ft_weight, "int16")

        for pa_layer in model.pa_input.get_layers():
            pa_bias = quantize_tensor(pa_layer.bias, scale, torch.int16)
            pa_weight = quantize_tensor(pa_layer.weight, scale, torch.int16)
            self._write_dense_block("pa", pa_bias, pa_weight, "int16")

    def write_fc_layer(
        self, model: model_lg.ReversiModel, layer: nn.Module, is_output: bool = False
    ) -> None:
        """Write a fully connected layer to the buffer."""
        weight_scale_hidden = model.weight_scale_hidden
        weight_scale_out = (
            model.score_scale * model.weight_scale_out / model.quantized_one
        )
        weight_scale = weight_scale_out if is_output else weight_scale_hidden
        bias_scale_out = model.weight_scale_out * model.score_scale
        bias_scale_hidden = model.weight_scale_hidden * model.quantized_one
        bias_scale = bias_scale_out if is_output else bias_scale_hidden

        with torch.no_grad():
            bias_tensor = layer.bias.detach().cpu()
            weight_tensor = layer.weight.detach().cpu()

        bias = quantize_tensor(bias_tensor, bias_scale, torch.int32)
        if is_output:
            weight = weight_tensor.mul(weight_scale).round().to(torch.int16)
        else:
            weight = (
                weight_tensor.clamp(-model.max_hidden_weight, model.max_hidden_weight)
                .mul(weight_scale)
                .round()
                .to(torch.int8)
            )

        num_input = weight.shape[1]
        if num_input % PADDING_ALIGNMENT != 0:
            pad_size = PADDING_ALIGNMENT - (num_input % PADDING_ALIGNMENT)
            weight = F.pad(weight, (0, pad_size), mode="constant", value=0)

        self._extend_tensor(bias, "int32")
        self._extend_tensor(weight, "int16" if is_output else "int8")

        if self.show_hist and is_output:
            print("Output layer parameters:")
            print(f"Weight: {weight.flatten()}")
            print(f"Bias: {bias.flatten()}")
