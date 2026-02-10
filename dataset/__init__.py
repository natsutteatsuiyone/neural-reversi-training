"""Fast C++ extension for Reversi feature dataset loading.

This module provides a high-performance IterableDataset implementation
using C++ for record parsing and feature generation.

Features:
    - Reads raw .bin files (24 bytes/record) directly
    - Real-time feature extraction with random symmetry augmentation
    - Multi-threaded batch prefetching for optimal GPU utilization
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import torch
from torch.utils.data import IterableDataset

from dataset import _C

__all__ = ["BinDataset", "custom_collate_fn"]

# Type alias for batch output
BatchTuple = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class BinDataset(IterableDataset[BatchTuple]):
    """High-performance Reversi dataset from raw .bin files.

    Reads raw .bin files (24 bytes/record) and generates features in real-time,
    applying random symmetry transformations for data augmentation.

    The dataset yields batches of ``(scores, features, mobility, ply)`` tensors:
        - ``scores``: float32 evaluation scores
        - ``features``: int64 pattern indices
        - ``mobility``: int64 legal move count
        - ``ply``: int64 move number

    Args:
        filepaths: List of paths to .bin data files.
        batch_size: Number of records per batch.
        file_usage_ratio: Fraction of files to use per epoch (0.0, 1.0].
        shuffle: Whether to shuffle files before reading. Defaults to True.
        prefetch_factor: Number of batches to prefetch in background. Defaults to 4.
        decompress_workers: Number of C++ worker threads for processing.
            If 0 (default), automatically selects based on hardware concurrency.
        seed: Random seed for reproducibility. If 0 (default), uses random device.

    Raises:
        ValueError: If any argument is out of valid range.
        FileNotFoundError: If any data file does not exist.

    Example:
        >>> dataset = BinDataset(["data.bin"], batch_size=1024)
        >>> for scores, features, mobility, ply in dataset:
        ...     pass  # Training loop
    """

    def __init__(
        self,
        filepaths: list[str],
        batch_size: int,
        file_usage_ratio: float,
        shuffle: bool = True,
        prefetch_factor: int = 4,
        decompress_workers: int = 0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self._validate_args(
            filepaths,
            batch_size,
            file_usage_ratio,
            prefetch_factor,
            decompress_workers,
        )

        self.filepaths = list(filepaths)
        self.batch_size = batch_size
        self.file_usage_ratio = file_usage_ratio
        self.shuffle = shuffle
        self.decompress_workers = decompress_workers
        self.prefetch_factor = prefetch_factor
        self.seed = seed
        self._epoch = 0
        self._reader: _C.BinDatasetReader | None = None

    @staticmethod
    def _validate_args(
        filepaths: list[str],
        batch_size: int,
        file_usage_ratio: float,
        prefetch_factor: int,
        decompress_workers: int,
    ) -> None:
        """Validate constructor arguments."""
        if not filepaths:
            raise ValueError("filepaths cannot be empty")

        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        if not (0.0 < file_usage_ratio <= 1.0):
            raise ValueError(
                f"file_usage_ratio must be in (0.0, 1.0], got {file_usage_ratio}"
            )

        if prefetch_factor < 1:
            raise ValueError(f"prefetch_factor must be >= 1, got {prefetch_factor}")

        if decompress_workers < 0:
            raise ValueError(
                f"decompress_workers must be >= 0, got {decompress_workers}"
            )

        # Validate that all files exist
        for fp in filepaths:
            if not os.path.exists(fp):
                raise FileNotFoundError(f"Data file not found: {fp}")

    def __iter__(self) -> Iterator[BatchTuple]:
        """Create a new iterator over the dataset.

        Returns:
            Iterator yielding (scores, features, mobility, ply) tensor tuples.
        """
        epoch_seed = (self.seed + self._epoch) if self.seed != 0 else 0
        self._epoch += 1

        self._reader = _C.BinDatasetReader(
            self.filepaths,
            self.batch_size,
            self.file_usage_ratio,
            self.shuffle,
            self.decompress_workers,
            self.prefetch_factor,
            epoch_seed,
        )

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            self._reader.set_worker_info(worker_info.id, worker_info.num_workers)

        return self

    def __next__(self) -> BatchTuple:
        """Get the next batch from the dataset.

        Returns:
            Tuple of (scores, features, mobility, ply) tensors.

        Raises:
            StopIteration: When all data has been consumed.
        """
        if self._reader is None:
            raise StopIteration

        result = self._reader.next()
        if result is None:
            raise StopIteration

        return result


def custom_collate_fn(batch: list[BatchTuple]) -> BatchTuple:
    """Identity collate function for DataLoader with batch_size=1.

    Since BinDataset already produces batched tensors, this function simply
    extracts the single batch from the DataLoader's wrapper list.

    Args:
        batch: List containing exactly one BatchTuple.

    Returns:
        The unwrapped BatchTuple.

    Raises:
        AssertionError: If batch does not contain exactly one element.
    """
    assert len(batch) == 1, "Expected batch size of 1 from IterableDataset"
    return batch[0]
