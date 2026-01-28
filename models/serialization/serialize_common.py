"""Base infrastructure for neural network serialization.

This module provides shared utilities and base classes used by all model-specific
serializers (large, small, WASM). It includes:
- Tensor quantization and conversion utilities
- Histogram visualization for debugging
- Checkpoint loading with EMA support
- Compressed output writing
- Abstract base class for NNWriter implementations
"""

from __future__ import annotations

import argparse
import sys
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

import numpy as np
import numpy.typing as npt
import torch
import zstandard as zstd

if TYPE_CHECKING:
    import lightning as L

# Constants
PADDING_ALIGNMENT = 32  # Neural network weight alignment requirement
DEFAULT_COMPRESSION_LEVEL = 1

# Histogram formatting defaults
DEFAULT_HISTOGRAM_BINS = 10
HISTOGRAM_WIDTH = 50
BIN_RANGE_WIDTH = 20
COUNT_WIDTH = 6
FLOAT_PRECISION = 4

ArrayLike = Union[Sequence[float], npt.NDArray[np.float64], torch.Tensor]


# =============================================================================
# Tensor utilities
# =============================================================================


def normalize_state_dict_keys(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Ensure checkpoint keys follow the ``model.*`` Lightning convention."""
    if any(key.startswith("model.") for key in state.keys()):
        return dict(state)
    return {f"model.{key}": value for key, value in state.items()}


def quantize_tensor(
    tensor: torch.Tensor, scale: float, dtype: torch.dtype
) -> torch.Tensor:
    """Quantize a tensor with the given scale and target dtype."""
    with torch.no_grad():
        return tensor.detach().cpu().mul(scale).round().to(dtype)


def tensor_to_little_endian_bytes(tensor: torch.Tensor, dtype: npt.DTypeLike) -> bytes:
    """Convert a tensor to bytes with a guaranteed little-endian representation."""
    np_dtype = np.dtype(dtype).newbyteorder("<")
    array = tensor.detach().cpu().contiguous().view(-1).numpy()
    return array.astype(np_dtype, copy=False).tobytes()


# =============================================================================
# Histogram utilities
# =============================================================================


def _to_numpy(data: ArrayLike) -> npt.NDArray[np.float64]:
    """Convert supported array-like inputs to a NumPy array."""
    if isinstance(data, torch.Tensor):
        array = data.detach().cpu().flatten().numpy()
    elif isinstance(data, np.ndarray):
        array = data.flatten()
    elif isinstance(data, Sequence):
        array = np.asarray(list(data), dtype=np.float64)
    else:
        raise TypeError(f"Unsupported histogram input type: {type(data)}")

    return array.astype(np.float64, copy=False)


def ascii_hist(name: str, data: ArrayLike, bins: int = DEFAULT_HISTOGRAM_BINS) -> None:
    """Display an ASCII histogram of the provided data."""
    if bins <= 0:
        raise ValueError(f"Number of bins must be positive, got {bins}")

    values = _to_numpy(data)
    if values.size == 0:
        print(f"{name}\nNo data provided.")
        return

    histogram_counts, bin_edges = np.histogram(values, bins=bins)
    max_count = histogram_counts.max() if histogram_counts.max() != 0 else 1

    print(name)
    for idx, count in enumerate(histogram_counts):
        bin_range = (
            f"[{bin_edges[idx]:.{FLOAT_PRECISION}g}, "
            f"{bin_edges[idx + 1]:.{FLOAT_PRECISION}g})"
        ).ljust(BIN_RANGE_WIDTH)
        bar = "#" * int(count * HISTOGRAM_WIDTH / max_count)
        formatted_count = f"({count:d})".rjust(COUNT_WIDTH)
        print(f"{bin_range}| {bar} {formatted_count}")


def maybe_ascii_hist(
    name: str,
    data: ArrayLike,
    *,
    show: bool,
    bins: int = DEFAULT_HISTOGRAM_BINS,
) -> None:
    """Render an ASCII histogram if ``show`` is True."""
    if not show:
        return
    ascii_hist(name, data, bins=bins)


# =============================================================================
# CLI and I/O utilities
# =============================================================================


def create_serialization_parser(
    description: str,
    default_filename: str | None = None,
) -> argparse.ArgumentParser:
    """Create a standard argument parser for serialization scripts.

    Args:
        description: Description text for the CLI.
        default_filename: Default output filename (e.g., "eval-{hash}.zst").

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(description=description)
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
        default=default_filename,
    )
    parser.add_argument(
        "--no-hist",
        action="store_true",
        help="Disable histogram display during serialization",
    )
    return parser


def load_checkpoint_into_model(
    checkpoint_path: str,
    lit_model: L.LightningModule,
    parser: argparse.ArgumentParser,
) -> None:
    """Load checkpoint weights into a LightningModule.

    Handles both EMA-averaged checkpoints and regular checkpoints,
    with key normalization for backward compatibility.

    Args:
        checkpoint_path: Path to the .ckpt file.
        lit_model: LightningModule instance to load weights into.
        parser: ArgumentParser for error reporting.
    """
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    except FileNotFoundError:
        parser.error(f"Checkpoint file not found: {checkpoint_path}")
    except Exception as exc:
        parser.error(f"Failed to load checkpoint: {exc}")

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


def write_compressed_output(
    data: bytes,
    output_path: Path,
    compression_level: int,
) -> None:
    """Write data to a zstd-compressed file.

    Args:
        data: Raw bytes to compress and write.
        output_path: Destination file path.
        compression_level: Zstandard compression level (1-22).
    """
    cctx = zstd.ZstdCompressor(level=compression_level)
    compressed_data = cctx.compress(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_path, "wb") as f:
            f.write(compressed_data)
        print(f"Model serialized to: {output_path}")
    except IOError as exc:
        print(f"Error writing output file: {exc}", file=sys.stderr)
        sys.exit(1)


# =============================================================================
# Base writer class
# =============================================================================


class BaseNNWriter(ABC):
    """Abstract base class for neural network serialization writers.

    Subclasses must implement write_model() to handle model-specific
    serialization logic.
    """

    buf: bytearray
    show_hist: bool

    def __init__(self, show_hist: bool = True) -> None:
        """Initialize the writer.

        Args:
            show_hist: Whether to display histograms during serialization.
        """
        self.buf = bytearray()
        self.show_hist = show_hist

    @abstractmethod
    def write_model(self, model: Any) -> None:
        """Write the model to the internal buffer.

        Args:
            model: The neural network model to serialize.
        """
        pass

    def get_buffer(self) -> bytes:
        """Return the serialized buffer."""
        return bytes(self.buf)

    def _extend_tensor(self, tensor: torch.Tensor, dtype: str) -> None:
        """Append tensor bytes to the buffer."""
        self.buf.extend(tensor_to_little_endian_bytes(tensor, dtype))

    def _write_dense_block(
        self,
        prefix: str,
        bias: torch.Tensor,
        weight: torch.Tensor,
        dtype: str,
    ) -> None:
        """Write a dense layer block with uniform dtype for bias and weight.

        Args:
            prefix: Label prefix for histogram display.
            bias: Bias tensor.
            weight: Weight tensor.
            dtype: Data type string for both tensors (e.g., "int16").
        """
        maybe_ascii_hist(f"{prefix} bias:", bias, show=self.show_hist)
        maybe_ascii_hist(f"{prefix} weight:", weight, show=self.show_hist)
        self._extend_tensor(bias, dtype)
        self._extend_tensor(weight, dtype)

    def _write_dense_block_mixed(
        self,
        prefix: str,
        bias: torch.Tensor,
        bias_dtype: str,
        weight: torch.Tensor,
        weight_dtype: str,
    ) -> None:
        """Write a dense layer block with different dtypes for bias and weight.

        Args:
            prefix: Label prefix for histogram display.
            bias: Bias tensor.
            bias_dtype: Data type string for bias (e.g., "int16").
            weight: Weight tensor.
            weight_dtype: Data type string for weight (e.g., "int8").
        """
        maybe_ascii_hist(f"{prefix} bias:", bias, show=self.show_hist)
        maybe_ascii_hist(f"{prefix} weight:", weight, show=self.show_hist)
        self._extend_tensor(bias, bias_dtype)
        self._extend_tensor(weight, weight_dtype)


__all__ = [
    # Constants
    "PADDING_ALIGNMENT",
    "DEFAULT_COMPRESSION_LEVEL",
    "DEFAULT_HISTOGRAM_BINS",
    # Tensor utilities
    "normalize_state_dict_keys",
    "quantize_tensor",
    "tensor_to_little_endian_bytes",
    # Histogram utilities
    "ascii_hist",
    "maybe_ascii_hist",
    # CLI and I/O
    "create_serialization_parser",
    "load_checkpoint_into_model",
    "write_compressed_output",
    # Base class
    "BaseNNWriter",
]
