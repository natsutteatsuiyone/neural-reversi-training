"""Neural network model serialization module.

This module provides functionality to serialize ReversiModel neural networks
into a compressed binary format suitable for deployment. The serialization
includes quantization of weights and biases to reduce model size.

Classes:
    NNWriter: Handles the serialization of neural network models.

Functions:
    ascii_hist: Display ASCII histogram for data visualization.
    main: CLI entry point for model serialization.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Union

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
import zstandard as zstd

import model as M
import version

# Constants
PADDING_ALIGNMENT = 32  # Neural network weight alignment requirement
DEFAULT_COMPRESSION_LEVEL = 7
DEFAULT_HISTOGRAM_BINS = 10
HISTOGRAM_WIDTH = 50
BIN_RANGE_WIDTH = 20
COUNT_WIDTH = 6
FLOAT_PRECISION = 4


def ascii_hist(
    name: str,
    x: Union[List[float], npt.NDArray[np.float64]],
    bins: int = DEFAULT_HISTOGRAM_BINS
) -> None:
    """Display an ASCII histogram of the data.

    Args:
        name: Title for the histogram.
        x: Data to visualize.
        bins: Number of bins for the histogram.

    Raises:
        ValueError: If bins <= 0.
        TypeError: If x is not a list or numpy array.
    """
    if bins <= 0:
        raise ValueError(f"Number of bins must be positive, got {bins}")

    if not isinstance(x, (list, np.ndarray)):
        raise TypeError(f"Expected list or numpy array, got {type(x)}")

    if len(x) == 0:
        print(f"{name}\nNo data provided.")
        return

    histogram_counts, bin_edges = np.histogram(x, bins=bins)
    max_count = histogram_counts.max() if histogram_counts.max() != 0 else 1

    print(name)
    for i in range(len(histogram_counts)):
        bin_range = f"[{bin_edges[i]:.{FLOAT_PRECISION}g}, {bin_edges[i + 1]:.{FLOAT_PRECISION}g})".ljust(BIN_RANGE_WIDTH)
        bar = "#" * int(histogram_counts[i] * HISTOGRAM_WIDTH / max_count)
        count = f"({histogram_counts[i]:d})".rjust(COUNT_WIDTH)
        print(f"{bin_range}| {bar} {count}")


def quantize_tensor(
    tensor: torch.Tensor,
    scale: float,
    dtype: torch.dtype
) -> torch.Tensor:
    """Quantize a tensor with the given scale and convert to specified dtype.

    Args:
        tensor: Input tensor to quantize.
        scale: Scale factor for quantization.
        dtype: Target data type.

    Returns:
        Quantized tensor.
    """
    with torch.no_grad():
        return tensor.cpu().mul(scale).round().to(dtype)


class NNWriter:
    """Serializes ReversiModel to a binary format with quantization.

    The serialization format (little-endian):
    - Input layer: bias (int16) + weight (int16)
    - Hidden layers: bias (int32) + weight (int8, padded to 32-byte alignment)
    - Output layer: bias (int32) + weight (int8)

    Attributes:
        buf: Binary buffer containing serialized data.
        show_hist: Whether to display histograms during serialization.
    """

    buf: bytearray
    show_hist: bool

    def __init__(self, model: M.ReversiModel, show_hist: bool = True) -> None:
        """Initialize the NNWriter.

        Args:
            model: The ReversiModel to serialize.
            show_hist: Whether to display histograms during serialization.
        """
        self.buf = bytearray()
        self.show_hist = show_hist

        self.write_input_layer(model)
        for (
            l1_base,
            l1_ps,
            l2,
            output,
        ) in model.layer_stacks.get_coalesced_layer_stacks():
            self.write_fc_layer(model, l1_base)
            self.write_fc_layer(model, l1_ps)
            self.write_fc_layer(model, l2)
            self.write_fc_layer(model, output, is_output=True)


    def get_buffer(self) -> bytes:
        """Return the serialized buffer.

        Returns:
            The serialized model data as bytes.
        """
        return bytes(self.buf)

    def write_input_layer(self, model: M.ReversiModel) -> None:
        """Write the input layer to the buffer.

        Args:
            model: The ReversiModel containing the input layer.
        """
        layer = model.base_input
        bias = quantize_tensor(layer.bias, model.quantized_one, torch.int16)
        weight = quantize_tensor(layer.weight, model.quantized_one, torch.int16)

        if self.show_hist:
            ascii_hist("ft bias:", bias.numpy())
            ascii_hist("ft weight:", weight.numpy())

        # Ensure little-endian byte order
        self.buf.extend(bias.flatten().numpy().astype('<i2').tobytes())
        self.buf.extend(weight.flatten().numpy().astype('<i2').tobytes())

        for layer in model.pa_input.get_layers():
            bias = quantize_tensor(layer.bias, model.quantized_one, torch.int16)
            weight = quantize_tensor(layer.weight, model.quantized_one, torch.int16)

            # Ensure little-endian byte order
            self.buf.extend(bias.flatten().numpy().astype('<i2').tobytes())
            self.buf.extend(weight.flatten().numpy().astype('<i2').tobytes())

    def write_fc_layer(
        self, model: M.ReversiModel, layer: nn.Module, is_output: bool = False
    ) -> None:
        """Write a fully connected layer to the buffer.

        Args:
            model: The ReversiModel for accessing quantization parameters.
            layer: The layer to serialize.
            is_output: Whether this is the output layer.
        """
        kWeightScaleHidden = model.weight_scale_hidden
        kWeightScaleOut = (
            model.score_scale * model.weight_scale_out / model.quantized_one
        )
        kWeightScale = kWeightScaleOut if is_output else kWeightScaleHidden
        kBiasScaleOut = model.weight_scale_out * model.score_scale
        kBiasScaleHidden = model.weight_scale_hidden * model.quantized_one
        kBiasScale = kBiasScaleOut if is_output else kBiasScaleHidden
        kMaxWeight = model.quantized_one / kWeightScale

        with torch.no_grad():
            bias_tensor = layer.bias.cpu()
            weight_tensor = layer.weight.cpu()

        bias = quantize_tensor(bias_tensor, kBiasScale, torch.int32)
        weight = weight_tensor
        clipped_diff = weight.clamp(-kMaxWeight, kMaxWeight) - weight
        clipped = torch.count_nonzero(clipped_diff)
        total_elements = torch.numel(weight)
        clipped_max = torch.max(torch.abs(clipped_diff)).item()

        weight = (
            weight.clamp(-kMaxWeight, kMaxWeight)
            .mul(kWeightScale)
            .round()
            .to(torch.int8)
        )

        if self.show_hist:
            print(
                f"Layer has {clipped}/{total_elements} clipped weights. "
                f"Maximum excess: {clipped_max} (limit: {kMaxWeight})."
            )

        num_input = weight.shape[1]
        if num_input % PADDING_ALIGNMENT != 0:
            pad_size = PADDING_ALIGNMENT - (num_input % PADDING_ALIGNMENT)
            padded_num = num_input + pad_size
            if self.show_hist:
                print(f"Padding input from {num_input} to {padded_num} elements.")
            weight = F.pad(weight, (0, pad_size), mode='constant', value=0)

        # Ensure little-endian byte order
        self.buf.extend(bias.flatten().numpy().astype('<i4').tobytes())
        self.buf.extend(weight.flatten().numpy().astype('<i1').tobytes())

        if self.show_hist:
            if is_output:
                print("Output layer parameters:")
                print(f"Weight: {weight.flatten()}")
                print(f"Bias: {bias.flatten()}")
            print()

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
        model = M.ReversiModel.load_from_checkpoint(args.checkpoint)
    except FileNotFoundError:
        parser.error(f"Checkpoint file not found: {args.checkpoint}")
    except Exception as e:
        parser.error(f"Failed to load checkpoint: {e}")

    model.eval()

    writer = NNWriter(model, show_hist=not args.no_hist)
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
