import random
from typing import Iterable, List, Tuple

import numpy as np
import torch
import zstandard
import queue
import threading

from torch.utils.data import IterableDataset


class FeatureDataset(IterableDataset):

    def __init__(
        self,
        filepaths: List[str],
        batch_size: int,
        file_usage_ratio: float,
        num_features: int,
        num_feature_params: List[int],
        shuffle: bool = True,
        prefetch_factor: int = 2,
    ):
        super().__init__()

        if not (0.0 < file_usage_ratio <= 1.0):
            raise ValueError("file_fraction_per_epoch must be in the range (0.0, 1.0].")

        self.filepaths = list(filepaths)
        self.batch_size = batch_size
        self.file_usage_ratio = file_usage_ratio
        self.shuffle = shuffle
        self.prefetch_factor = prefetch_factor

        self.record_dtype = np.dtype(
            [
                ("score", np.float32, 1),
                ("features", np.uint16, num_features),
                ("mobility", np.uint8, 1),
                ("ply", np.uint8, 1),
            ]
        )
        self.record_size = self.record_dtype.itemsize
        self.batch_bytes = self.record_size * self.batch_size
        self.chunk_bytes = max(self.batch_bytes * 64, 1 << 20)

        self.feature_cum_offsets = np.cumsum([0] + num_feature_params[:-1], dtype=np.int64)

    def __iter__(self):
        all_paths = self.filepaths.copy()

        if self.shuffle:
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

        iterator = self._feature_iterator(selected_paths)
        if self.prefetch_factor > 0:
            return self._prefetched_feature_iterator(iterator)
        return iterator

    def _feature_iterator(self, paths: Iterable[str]):
        pending = b""

        for path in paths:
            dctx = zstandard.ZstdDecompressor()
            with open(path, "rb") as fp, dctx.stream_reader(fp) as reader:
                while True:
                    chunk = reader.read(self.chunk_bytes)
                    if not chunk:
                        break

                    if pending:
                        chunk = pending + chunk
                        pending = b""

                    total_records = len(chunk) // self.record_size
                    if total_records < self.batch_size:
                        pending = chunk
                        continue

                    full_batches = total_records // self.batch_size
                    cutoff = full_batches * self.batch_bytes
                    records = np.frombuffer(chunk[:cutoff], dtype=self.record_dtype)

                    for start in range(0, len(records), self.batch_size):
                        batch = records[start : start + self.batch_size]
                        yield self._process(batch)

                    pending = chunk[cutoff:]

        # Drop any trailing records that cannot form a full batch

    def _process(self, arr: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scores = torch.from_numpy(arr["score"].copy())

        features_raw = arr["features"]
        features_adjusted = features_raw.astype(np.int64)
        features_adjusted += self.feature_cum_offsets
        features = torch.from_numpy(features_adjusted)

        mobility = torch.from_numpy(arr["mobility"].astype(np.int64, copy=False))
        ply = torch.from_numpy(arr["ply"].astype(np.int64, copy=False))
        return scores, features, mobility, ply

    def _prefetched_feature_iterator(self, iterator: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
        sentinel = object()
        q: "queue.Queue[object]" = queue.Queue(maxsize=self.prefetch_factor)

        def producer():
            try:
                for item in iterator:
                    q.put(item)
            except Exception as exc:  # pragma: no cover - surfaced to consumer
                q.put(exc)
            finally:
                q.put(sentinel)

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()

        while True:
            item = q.get()
            if item is sentinel:
                thread.join()
                return
            if isinstance(item, Exception):
                thread.join()
                raise item
            yield item


def custom_collate_fn(batch):
    assert len(batch) == 1, "Expected batch size of 1 from IterableDataset"
    return batch[0]
