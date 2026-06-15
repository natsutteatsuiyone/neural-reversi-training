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
        for l1, l2, output in model.layer_stacks.get_layer_stacks():
            self.write_hidden_layer(
                model,
                l1,
                input_quantized_one=model.input_quantized_one,
                output_quantized_one=model.hidden_quantized_one,
            )
            self.write_hidden_layer(
                model,
                l2,
                input_quantized_one=model.hidden_quantized_one,
                output_quantized_one=model.hidden_quantized_one,
            )
            self.write_output_layer(model, output)

    def write_input_layer(self, model: model_wasm.ReversiWasmModel) -> None:
        """Write the input layer to the buffer."""
        bias = quantize_tensor(model.input.bias, model.input_quantized_one, torch.int16)
        weight = model.input.weight.clamp(
            -model.max_input_weight, model.max_input_weight
        )
        weight = quantize_tensor(weight, model.input_quantized_one, torch.int8)
        self._write_dense_block_mixed("input", bias, "int16", weight, "int8")

    def write_hidden_layer(
        self,
        model: model_wasm.ReversiWasmModel,
        layer: nn.Module,
        input_quantized_one: float,
        output_quantized_one: float,
    ) -> None:
        """Write an l1/l2 hidden layer to the buffer."""
        cfg = model.config
        weight_scale = (
            cfg.weight_scale_hidden * output_quantized_one / input_quantized_one
        )
        bias_scale = cfg.weight_scale_hidden * output_quantized_one

        with torch.no_grad():
            bias_tensor = layer.bias.detach().cpu()
            weight_tensor = layer.weight.detach().cpu()

        bias = quantize_tensor(bias_tensor, bias_scale, torch.int32)
        weight = self._quantize_i16_weight(weight_tensor, weight_scale)

        self._extend_tensor(bias, "int32")
        self._extend_tensor(weight, "int16")

    def write_output_layer(
        self,
        model: model_wasm.ReversiWasmModel,
        layer: nn.Module,
    ) -> None:
        """Write the output layer with mixed activation input scales."""
        cfg = model.config
        bias_scale = cfg.weight_scale_out * cfg.score_scale
        hidden_weight_scale = (
            cfg.score_scale * cfg.weight_scale_out / model.hidden_quantized_one
        )
        skip_weight_scale = (
            cfg.score_scale * cfg.weight_scale_out / model.input_quantized_one
        )

        with torch.no_grad():
            bias_tensor = layer.bias.detach().cpu()
            weight_tensor = layer.weight.detach().cpu()

        bias = quantize_tensor(bias_tensor, bias_scale, torch.int32)
        hidden_weight = self._quantize_i16_weight(
            weight_tensor[:, : model_wasm.LHIDDEN],
            hidden_weight_scale,
        )
        skip_weight = self._quantize_i16_weight(
            weight_tensor[:, model_wasm.LHIDDEN :],
            skip_weight_scale,
        )
        weight = torch.cat((hidden_weight, skip_weight), dim=1)

        self._extend_tensor(bias, "int32")
        self._extend_tensor(weight, "int16")

        if self.show_hist:
            print("Output layer parameters:")
            print(f"Weight: {weight.flatten()}")
            print(f"Bias: {bias.flatten()}")
            print()

    @staticmethod
    def _quantize_i16_weight(weight_tensor: torch.Tensor, scale: float) -> torch.Tensor:
        max_weight = torch.iinfo(torch.int16).max / scale
        return (
            weight_tensor.clamp(-max_weight, max_weight)
            .mul(scale)
            .round()
            .to(torch.int16)
        )
