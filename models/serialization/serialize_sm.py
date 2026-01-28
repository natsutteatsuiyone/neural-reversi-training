"""Small model (ReversiSmallModel) serialization writer."""

from typing import Any

import torch
import torch.nn as nn

from models import model_sm
from models.serialization.serialize_common import BaseNNWriter, quantize_tensor


class SmallNNWriter(BaseNNWriter):
    """Serialize ``ReversiSmallModel`` instances into the binary format."""

    def __init__(
        self, model: model_sm.ReversiSmallModel, show_hist: bool = True
    ) -> None:
        super().__init__(show_hist)
        self.write_model(model)

    def write_model(self, model: Any) -> None:
        """Write the small model to the internal buffer."""
        self.write_input_layer(model)
        for output in model.layer_stacks.get_layer_stacks():
            self.write_fc_layer(model, output, is_output=True)

    def write_input_layer(self, model: model_sm.ReversiSmallModel) -> None:
        """Write the input layer to the buffer."""
        for pa_layer in model.pa_input.get_layers():
            bias = quantize_tensor(pa_layer.bias, model.quantized_one, torch.int16)
            weight = quantize_tensor(pa_layer.weight, model.quantized_one, torch.int16)
            self._write_dense_block("pa", bias, weight, "int16")

    def write_fc_layer(
        self,
        model: model_sm.ReversiSmallModel,
        layer: nn.Module,
        is_output: bool = False,
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
        max_weight = model.quantized_one / weight_scale

        with torch.no_grad():
            bias_tensor = layer.bias.detach().cpu()
            weight_tensor = layer.weight.detach().cpu()

        bias = quantize_tensor(bias_tensor, bias_scale, torch.int32)

        if is_output:
            weight = weight_tensor.mul(weight_scale).round().to(torch.int16)
        else:
            weight = (
                weight_tensor.clamp(-max_weight, max_weight)
                .mul(weight_scale)
                .round()
                .to(torch.int8)
            )

        self._extend_tensor(bias, "int32")
        self._extend_tensor(weight, "int16" if is_output else "int8")

        if self.show_hist and is_output:
            print("Output layer parameters:")
            print(f"Weight: {weight.flatten()}")
            print(f"Bias: {bias.flatten()}")
            print()
