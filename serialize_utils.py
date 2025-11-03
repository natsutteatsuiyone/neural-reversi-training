"""Shared utilities for neural network serialization scripts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Union

import numpy as np
import numpy.typing as npt
import torch

# Histogram formatting defaults shared by serialize scripts.
DEFAULT_HISTOGRAM_BINS = 10
HISTOGRAM_WIDTH = 50
BIN_RANGE_WIDTH = 20
COUNT_WIDTH = 6
FLOAT_PRECISION = 4

ArrayLike = Union[Sequence[float], npt.NDArray[np.float64], torch.Tensor]


def normalize_state_dict_keys(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Ensure checkpoint keys follow the ``model.*`` Lightning convention."""
    if any(key.startswith("model.") for key in state.keys()):
        return dict(state)
    return {f"model.{key}": value for key, value in state.items()}


def quantize_tensor(tensor: torch.Tensor, scale: float, dtype: torch.dtype) -> torch.Tensor:
    """Quantize a tensor with the given scale and target dtype."""
    with torch.no_grad():
        return tensor.detach().cpu().mul(scale).round().to(dtype)


def tensor_to_little_endian_bytes(tensor: torch.Tensor, dtype: npt.DTypeLike) -> bytes:
    """Convert a tensor to bytes with a guaranteed little-endian representation."""
    np_dtype = np.dtype(dtype).newbyteorder("<")
    array = tensor.detach().cpu().contiguous().view(-1).numpy()
    return array.astype(np_dtype, copy=False).tobytes()


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


__all__ = [
    "DEFAULT_HISTOGRAM_BINS",
    "HISTOGRAM_WIDTH",
    "BIN_RANGE_WIDTH",
    "COUNT_WIDTH",
    "FLOAT_PRECISION",
    "ascii_hist",
    "maybe_ascii_hist",
    "normalize_state_dict_keys",
    "quantize_tensor",
    "tensor_to_little_endian_bytes",
]
