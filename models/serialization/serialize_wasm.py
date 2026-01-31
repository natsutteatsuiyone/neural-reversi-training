"""WASM model (ReversiWasmModel) serialization writer."""

from typing import Any

import torch
import torch.nn as nn

from models import model_wasm
from models.serialization.serialize_common import BaseNNWriter, quantize_tensor


class WasmNNWriter(BaseNNWriter):
    """Serialize ``ReversiWasmModel`` instances into the binary format."""

    def __init__(
        self, model: model_wasm.ReversiWasmModel, show_hist: bool = True
    ) -> None:
        super().__init__(show_hist)
        self.write_model(model)

    def write_model(self, model: Any) -> None:
        """Write the WASM model to the internal buffer."""
        self.write_input_layer(model)
        for output in model.layer_stacks.get_layer_stacks():
            self.write_fc_layer(model, output)

    def write_input_layer(self, model: model_wasm.ReversiWasmModel) -> None:
        """Write the input layer to the buffer."""
        bias = quantize_tensor(model.input.bias, model.quantized_one, torch.int16)
        weight = model.input.weight.clamp(
            -model.max_input_weight, model.max_input_weight
        )
        weight = quantize_tensor(weight, model.quantized_one, torch.int8)
        self._write_dense_block_mixed("input", bias, "int16", weight, "int8")

    def write_fc_layer(
        self,
        model: model_wasm.ReversiWasmModel,
        layer: nn.Module,
    ) -> None:
        """Write a fully connected layer to the buffer."""
        weight_scale_out = (
            model.score_scale * model.weight_scale_out / model.quantized_one
        )
        weight_scale = weight_scale_out
        bias_scale_out = model.weight_scale_out * model.score_scale
        bias_scale = bias_scale_out

        with torch.no_grad():
            bias_tensor = layer.bias.detach().cpu()
            weight_tensor = layer.weight.detach().cpu()

        bias = quantize_tensor(bias_tensor, bias_scale, torch.int32)
        weight = weight_tensor.mul(weight_scale).round().to(torch.int16)

        self._extend_tensor(bias, "int32")
        self._extend_tensor(weight, "int16")

        if self.show_hist:
            print("Output layer parameters:")
            print(f"Weight: {weight.flatten()}")
            print(f"Bias: {bias.flatten()}")
            print()
