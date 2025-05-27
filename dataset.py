import random
import numpy as np
import torch
import zstandard
from typing import List, Tuple

from torch.utils.data import IterableDataset


class FeatureDataset(IterableDataset):

    def __init__(
        self,
        filepaths: List[str],
        batch_size: int,
        file_usage_ratio: float,
        num_features: int,
        num_feature_params: List[int],
    ):
        super().__init__()

        if not (0.0 < file_usage_ratio <= 1.0):
            raise ValueError("file_fraction_per_epoch must be in the range (0.0, 1.0].")

        self.filepaths = list(filepaths)
        self.batch_size = batch_size
        self.file_usage_ratio = file_usage_ratio

        self.record_dtype = np.dtype(
            [
                ("score", np.float32, 1),
                ("features", np.uint16, num_features),
                ("mobility", np.uint8, 1),
                ("ply", np.uint8, 1),
            ]
        )

        self.feature_cum_offsets = np.cumsum([0] + num_feature_params[:-1], dtype=np.int64)

    def __iter__(self):
        all_paths = self.filepaths.copy()
        random.shuffle(all_paths)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            paths_for_worker = all_paths
        else:
            paths_for_worker = all_paths[worker_info.id :: worker_info.num_workers]

        if not paths_for_worker:
            return iter([])

        n_total = len(paths_for_worker)
        n_use = max(1, int(round(n_total * self.file_usage_ratio)))
        selected_paths = paths_for_worker[:n_use]

        return self._feature_iterator(selected_paths)

    def _feature_iterator(self, paths: List[str]):
        record_size = self.record_dtype.itemsize
        dctx = zstandard.ZstdDecompressor()
        chunk_bytes = record_size * self.batch_size * 10  # read ≈10 batches at once

        buffer = bytearray()
        batch_buf = np.empty((0,), dtype=self.record_dtype)

        for path in paths:
            with open(path, "rb") as fp, dctx.stream_reader(fp) as reader:
                while True:
                    chunk = reader.read(chunk_bytes)
                    if not chunk:
                        break

                    buffer.extend(chunk)
                    n_records = len(buffer) // record_size
                    if n_records == 0:
                        continue

                    valid_len = n_records * record_size
                    arr = np.frombuffer(memoryview(buffer)[:valid_len], dtype=self.record_dtype)
                    buffer = buffer[valid_len:]

                    batch_buf = np.concatenate((batch_buf, arr)) if batch_buf.size else arr

                    while batch_buf.shape[0] >= self.batch_size:
                        yield self._process(batch_buf[: self.batch_size])
                        batch_buf = batch_buf[self.batch_size :]

    def _process(self, arr: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores = torch.from_numpy(arr["score"].copy())
        features = torch.from_numpy((arr["features"] + self.feature_cum_offsets).copy())
        mobility = torch.from_numpy(arr["mobility"].copy())
        ply = torch.from_numpy(arr["ply"].copy())
        return scores, features, mobility, ply


def custom_collate_fn(batch):
    assert len(batch) == 1, "Expected batch size of 1 from IterableDataset"
    return batch[0]
