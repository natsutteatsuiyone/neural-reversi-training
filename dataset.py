from random import shuffle
import numpy as np
import torch
import zstandard
from typing import List, Tuple

from torch.utils.data import IterableDataset

NUM_FEATURE_PARAMS = [
    6561, 6561, 6561, 6561,
    6561, 6561, 6561, 6561,
    6561, 6561,
    6561, 6561, 6561, 6561,
]

NUM_FEATURES = len(NUM_FEATURE_PARAMS)
SUM_OF_FEATURES = sum(NUM_FEATURE_PARAMS)

class FeatureDataset(IterableDataset):
    """
    A dataset that reads zstandard-compressed feature files
    and iterates over them as PyTorch tensors in batches.

    Args:
        filepaths (List[str]): List of file paths to read.
        batch_size (int): Size of each batch.
        random_skipping (float): Parameter for random skipping.
            A higher value means more data is skipped (skip_prob = N / (N+1)).
            If 0, no skipping occurs.
    """
    def __init__(self, filepaths: List[str], batch_size: int, random_skipping: float):
        super().__init__()
        self.filepaths = filepaths
        self.batch_size = batch_size
        if random_skipping < 0:
            raise ValueError("random_skipping cannot be negative")
        self.random_skipping = random_skipping

        self.record_dtype = np.dtype(
            [
                ("score", np.float32),
                ("features", np.uint16, NUM_FEATURES),
                ("mobility", np.uint8),
                ("ply", np.uint8),
            ]
        )

        self.feature_cum_offsets = np.zeros(NUM_FEATURES, dtype=np.int64)
        for i in range(1, NUM_FEATURES):
            self.feature_cum_offsets[i] = (
                self.feature_cum_offsets[i - 1] + NUM_FEATURE_PARAMS[i - 1]
            )

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single-worker case
            filepaths_for_worker = self.filepaths
        else:
            # Multi-worker case: split files
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            filepaths_for_worker = self.filepaths[worker_id::num_workers]

        filepaths_to_iterate = list(filepaths_for_worker)
        shuffle(filepaths_to_iterate)
        return self._feature_iterator(filepaths_to_iterate)

    def _feature_iterator(self, filepaths: List[str]):
        record_size = self.record_dtype.itemsize
        dctx = zstandard.ZstdDecompressor()
        CHUNK_SIZE_FACTOR = 10
        chunk_size = record_size * self.batch_size * CHUNK_SIZE_FACTOR
        buffer = bytearray()
        batch_buffer = np.empty((0,), dtype=self.record_dtype)
        skip_prob = self.random_skipping / (self.random_skipping + 1.0)

        for filepath in filepaths:
            try:
                with open(filepath, "rb") as fin:
                    with dctx.stream_reader(fin) as reader:
                        while True:
                            try:
                                chunk = reader.read(chunk_size)
                            except zstandard.ZstdError as e:
                                print(f"Warning: Error reading chunk from {filepath}: {e}")
                                break
                            except Exception as e:
                                print(f"Warning: Unexpected error during reading {filepath}: {e}")

                            if not chunk:
                                break # End of file

                            buffer.extend(chunk)
                            num_full_records = len(buffer) // record_size

                            if num_full_records > 0:
                                valid_bytes_len = num_full_records * record_size
                                data_bytes = buffer[:valid_bytes_len]
                                buffer = buffer[valid_bytes_len:]

                                try:
                                    arr = np.frombuffer(data_bytes, dtype=self.record_dtype)
                                except ValueError as e:
                                     print(f"Warning: Error converting buffer to numpy array from {filepath}. Buffer length: {len(data_bytes)}, Record size: {record_size}. Error: {e}")
                                     continue


                                if skip_prob > 0:
                                    mask = np.random.rand(arr.shape[0]) >= skip_prob
                                    selected = arr[mask]
                                else:
                                    selected = arr

                                if selected.size > 0:
                                    if batch_buffer is None or batch_buffer.size == 0: # Check size if using np.empty
                                         batch_buffer = selected
                                    else:
                                         batch_buffer = np.concatenate((batch_buffer, selected))


                                while batch_buffer is not None and batch_buffer.shape[0] >= self.batch_size:
                                    batch = batch_buffer[:self.batch_size]
                                    yield self._process(batch)
                                    batch_buffer = batch_buffer[self.batch_size:]
                                    if batch_buffer.shape[0] == 0:
                                        batch_buffer = np.empty((0,), dtype=self.record_dtype)

            except FileNotFoundError:
                print(f"Warning: File not found {filepath}, skipping.")
                continue
            except zstandard.ZstdError as e:
                 print(f"Warning: zstandard error opening or processing {filepath}: {e}")
                 continue
            except Exception as e:
                print(f"Warning: An unexpected error occurred processing {filepath}: {e}")
                continue

        if batch_buffer is not None and batch_buffer.shape[0] >= self.batch_size:
            batch = batch_buffer[:self.batch_size]
            yield self._process(batch)


    def _process(self, arr: np.ndarray, debug: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Converts a batch of Numpy arrays to PyTorch tensors.
           Includes .copy() to handle potential stride issues with from_numpy.
        """
        features_with_offset = arr["features"] + self.feature_cum_offsets
        feature_indices = torch.from_numpy(features_with_offset.copy())
        mobility = torch.from_numpy(arr["mobility"].copy())
        scores = torch.from_numpy(arr["score"].copy())
        ply = torch.from_numpy(arr["ply"].copy())

        if debug:
            print(feature_indices, mobility, scores, ply)
        return feature_indices, mobility, scores, ply


def custom_collate_fn(batch):
    """
    Custom collate function for FeatureDataset.
    Since the Dataset already yields pre-formed batches, this function
    simply returns the first (and only) element of the input list `batch`.
    """
    assert len(batch) == 1
    return batch[0]
