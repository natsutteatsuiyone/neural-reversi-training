"""Neural network model serialization module.

This module provides functionality to serialize ReversiModel neural networks
into a compressed binary format suitable for deployment. The serialization
includes quantization of weights and biases to reduce model size.

Classes:
    NNWriter: Handles the serialization of neural network models.

Functions:
    main: CLI entry point for model serialization.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import zstandard as zstd

import model as M
import version
from serialize_utils import (
    maybe_ascii_hist,
    normalize_state_dict_keys,
    quantize_tensor,
    tensor_to_little_endian_bytes,
)

# Constants
PADDING_ALIGNMENT = 32  # Neural network weight alignment requirement
DEFAULT_COMPRESSION_LEVEL = 1


class NNWriter:
    """Serialize ``ReversiModel`` instances into the binary NNUE format."""

    buf: bytearray
    show_hist: bool

    def __init__(self, model: M.ReversiModel, show_hist: bool = True) -> None:
        self.buf = bytearray()
        self.show_hist = show_hist

        self.write_input_layer(model)
        for (
            l1_base,
            l1_ps,
            l2,
            output,
        ) in model.layer_stacks.get_layer_stacks():
            self.write_fc_layer(model, l1_base)
            self.write_fc_layer(model, l1_ps)
            self.write_fc_layer(model, l2)
            self.write_fc_layer(model, output, is_output=True)

    def get_buffer(self) -> bytes:
        """Return the serialized buffer."""
        return bytes(self.buf)

    def write_input_layer(self, model: M.ReversiModel) -> None:
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

    def _write_dense_block(
        self,
        prefix: str,
        bias: torch.Tensor,
        weight: torch.Tensor,
        dtype: str,
    ) -> None:
        maybe_ascii_hist(f"{prefix} bias:", bias, show=self.show_hist)
        maybe_ascii_hist(f"{prefix} weight:", weight, show=self.show_hist)
        self._extend_tensor(bias, dtype)
        self._extend_tensor(weight, dtype)

    def write_fc_layer(
        self, model: M.ReversiModel, layer: nn.Module, is_output: bool = False
    ) -> None:
        """Write a fully connected layer to the buffer."""
        weight_scale_hidden = model.weight_scale_hidden
        weight_scale_out = model.score_scale * model.weight_scale_out / model.quantized_one
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

    def _extend_tensor(self, tensor: torch.Tensor, dtype: str) -> None:
        self.buf.extend(tensor_to_little_endian_bytes(tensor, dtype))

def main() -> None:
    """CLI entry point for model serialization."""
    parser = argparse.ArgumentParser(
        description="Serialize a ReversiModel to compressed binary format."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--cl",
        type=int,
        default=DEFAULT_COMPRESSION_LEVEL,
        help=f"Compression level (default: {DEFAULT_COMPRESSION_LEVEL})"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
    )
    parser.add_argument(
        "--no-hist",
        action="store_true",
        help="Disable histogram display during serialization"
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=f"eval-{version.get_version_hash()}.zst",
    )
    args = parser.parse_args()

    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
    except FileNotFoundError:
        parser.error(f"Checkpoint file not found: {args.checkpoint}")
    except Exception as e:
        parser.error(f"Failed to load checkpoint: {e}")

    # Instantiate LitReversiModel and load weights with backward-compat mapping
    lit_model = M.LitReversiModel()

    has_ema_metadata = isinstance(ckpt, dict) and any(
        key in ckpt for key in ("averaging_state", "current_model_state")
    )

    if isinstance(ckpt, dict) and "current_model_state" in ckpt:
        base_state = normalize_state_dict_keys(ckpt["current_model_state"])
        lit_model.load_state_dict(base_state, strict=False)

    state_source: Any
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_source = ckpt["state_dict"]
    else:
        state_source = ckpt

    state = normalize_state_dict_keys(state_source)
    _, unexpected = lit_model.load_state_dict(state, strict=False)
    if has_ema_metadata:
        print("Using EMA-averaged weights from checkpoint.", flush=True)
    if unexpected:
        print(f"Warning: unexpected keys in checkpoint: {sorted(unexpected)[:5]}...", flush=True)

    lit_model.eval()

    # Use the core model for serialization
    writer = NNWriter(lit_model.model, show_hist=not args.no_hist)
    cctx = zstd.ZstdCompressor(level=args.cl)
    compressed_data = cctx.compress(writer.get_buffer())

    output_filename = args.filename
    output_path = Path(args.output_dir) / output_filename

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_path, "wb") as f:
            f.write(compressed_data)
        print(f"Model serialized to: {output_path}")
    except IOError as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
