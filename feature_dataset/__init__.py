"""Fast C++ extension for Reversi feature dataset loading.

This module provides a high-performance IterableDataset implementation
using C++ for zstd decompression and record parsing.
"""

from __future__ import annotations

from typing import Iterator, List, Tuple

import torch
from torch.utils.data import IterableDataset

from feature_dataset import _C


class FeatureDataset(IterableDataset):
    """High-performance Reversi feature dataset using C++ backend.

    This is a drop-in replacement for training.dataset.FeatureDataset
    with significantly improved data loading performance.

    Args:
        filepaths: List of paths to zstd-compressed data files.
        batch_size: Number of records per batch.
        file_usage_ratio: Fraction of files to use per epoch (0.0, 1.0].
        num_features: Number of features per record (must be 24).
        num_feature_params: List of feature dimensions (used for validation).
        shuffle: Whether to shuffle files before reading.
        prefetch_factor: Number of batches to prefetch in background (default: 4).
        decompress_workers: Number of C++ worker threads for decompression (default: 2).
    """

    def __init__(
        self,
        filepaths: List[str],
        batch_size: int,
        file_usage_ratio: float,
        num_features: int,
        num_feature_params: List[int],
        shuffle: bool = True,
        prefetch_factor: int = 4,
        decompress_workers: int = 2,
    ):
        super().__init__()

        if num_features != 24:
            raise ValueError(f"num_features must be 24, got {num_features}")

        if not (0.0 < file_usage_ratio <= 1.0):
            raise ValueError("file_usage_ratio must be in the range (0.0, 1.0].")

        self.filepaths = list(filepaths)
        self.batch_size = batch_size
        self.file_usage_ratio = file_usage_ratio
        self.shuffle = shuffle
        self.decompress_workers = decompress_workers
        self.prefetch_factor = prefetch_factor
        self._reader: _C.FeatureDatasetReader | None = None

    def __iter__(
        self,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        self._reader = _C.FeatureDatasetReader(
            self.filepaths,
            self.batch_size,
            self.file_usage_ratio,
            self.shuffle,
            self.decompress_workers,
            self.prefetch_factor,
        )

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            self._reader.set_worker_info(worker_info.id, worker_info.num_workers)

        return self

    def __next__(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._reader is None:
            raise StopIteration

        result = self._reader.next()
        if result is None:
            raise StopIteration

        return result


def custom_collate_fn(batch):
    """Collate function for DataLoader with batch_size=1."""
    assert len(batch) == 1, "Expected batch size of 1 from IterableDataset"
    return batch[0]
