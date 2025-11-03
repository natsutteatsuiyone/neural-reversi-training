from typing import List

import torch

# Shared feature configuration used by both full and small models.
NUM_FEATURE_PARAMS: List[int] = [
    6561, 6561, 6561, 6561,
    6561, 6561, 6561, 6561,
    6561, 6561, 6561, 6561,
    6561, 6561, 6561, 6561,
    6561, 6561, 6561, 6561,
    19683, 19683, 19683, 19683,
]

NUM_FEATURES = len(NUM_FEATURE_PARAMS)
SUM_OF_FEATURES = sum(NUM_FEATURE_PARAMS)


def repeat_first_block(param: torch.Tensor, block_size: int, repeat: int, dim: int = 0) -> None:
    """Copy the first block along ``dim`` into all subsequent blocks in-place."""
    if repeat <= 1:
        return

    first_block = param.narrow(dim, 0, block_size).clone()
    for idx in range(1, repeat):
        param.narrow(dim, idx * block_size, block_size).copy_(first_block)


def select_bucket(flat_source: torch.Tensor, feature_dim: int, bucket_indices: torch.Tensor) -> torch.Tensor:
    """Slice the flattened tensor so each batch pulls the row matching its bucket."""
    return flat_source.reshape(-1, feature_dim).index_select(0, bucket_indices)


def bucket_lookup_indices(ply: torch.Tensor, bucket_size: int, count: int) -> torch.Tensor:
    """Return flattened indices that map each batch item to its phase bucket."""
    flat_ply = ply.view(-1)
    batch_size = flat_ply.size(0)
    ls_indices = flat_ply // bucket_size
    offsets = torch.arange(batch_size, device=flat_ply.device, dtype=torch.long) * count
    return ls_indices + offsets
