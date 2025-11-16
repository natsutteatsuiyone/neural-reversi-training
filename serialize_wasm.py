import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import zstandard as zstd

import model_wasm as M
import version
from serialize_utils import (
    maybe_ascii_hist,
    normalize_state_dict_keys,
    quantize_tensor,
    tensor_to_little_endian_bytes,
)

DEFAULT_COMPRESSION_LEVEL = 1


class NNWriter:
    buf: bytearray
    show_hist: bool

    def __init__(self, model: M.ReversiWasmlModel, show_hist: bool = True) -> None:
        self.buf = bytearray()
        self.show_hist = show_hist

        self.write_input_layer(model)
        for output in model.layer_stacks.get_layer_stacks():
            self.write_fc_layer(model, output)

    def get_buffer(self) -> bytes:
        return bytes(self.buf)

    def write_input_layer(self, model: M.ReversiWasmlModel) -> None:
        bias = quantize_tensor(model.input.bias, model.quantized_one, torch.int16)
        weight = model.input.weight.clamp(-model.max_input_weight, model.max_input_weight)
        weight = quantize_tensor(weight, model.quantized_one, torch.int8)
        self._write_dense_block("input", bias, "int16", weight, "int8")

    def _write_dense_block(
        self,
        prefix: str,
        bias: torch.Tensor,
        bias_dtype: str,
        weight: torch.Tensor,
        weight_dtype: str,
    ) -> None:
        maybe_ascii_hist(f"{prefix} bias:", bias, show=self.show_hist)
        maybe_ascii_hist(f"{prefix} weight:", weight, show=self.show_hist)
        self._extend_tensor(bias, bias_dtype)
        self._extend_tensor(weight, weight_dtype)

    def write_fc_layer(
        self,
        model: M.ReversiWasmlModel,
        layer: nn.Module,
    ) -> None:
        weight_scale_out = model.score_scale * model.weight_scale_out / model.quantized_one
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

    def _extend_tensor(self, tensor: torch.Tensor, dtype: str) -> None:
        self.buf.extend(tensor_to_little_endian_bytes(tensor, dtype))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serialize a ReversiWasmlModel to compressed binary format."
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
        help=f"Compression level (default: {DEFAULT_COMPRESSION_LEVEL})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=f"eval_wasm-{version.get_version_hash()}.zst",
    )
    parser.add_argument(
        "--no-hist",
        action="store_true",
        help="Disable histogram display during serialization",
    )
    args = parser.parse_args()

    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
    except FileNotFoundError:
        parser.error(f"Checkpoint file not found: {args.checkpoint}")
    except Exception as exc:
        parser.error(f"Failed to load checkpoint: {exc}")

    lit_model = M.LitReversiWasmModel()

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
        print(
            f"Warning: unexpected keys in checkpoint: {sorted(unexpected)[:5]}...",
            flush=True,
        )

    lit_model.eval()

    writer = NNWriter(lit_model.model, show_hist=not args.no_hist)
    cctx = zstd.ZstdCompressor(level=args.cl)
    compressed_data = cctx.compress(writer.get_buffer())

    output_path = Path(args.output_dir) / args.filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_path, "wb") as f:
            f.write(compressed_data)
        print(f"Model serialized to: {output_path}")
    except IOError as exc:
        print(f"Error writing output file: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
