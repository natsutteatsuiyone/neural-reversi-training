import random
import numpy as np
import torch
import zstandard
from typing import List, Tuple
import queue
import threading
import time

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

        if self.prefetch_factor > 0:
            return self._prefetched_feature_iterator(selected_paths)
        else:
            return self._feature_iterator(selected_paths)

    def _feature_iterator(self, paths: List[str]):
        record_size = self.record_dtype.itemsize
        dctx = zstandard.ZstdDecompressor()
        chunk_bytes = record_size * self.batch_size * 20  # Larger chunk size

        # Pre-allocate larger buffers
        buffer = bytearray(chunk_bytes * 4)
        buffer_view = memoryview(buffer)
        buffer_len = 0
        batch_buf = np.empty((self.batch_size * 4,), dtype=self.record_dtype)
        batch_buf_len = 0

        for path in paths:
            with open(path, "rb") as fp, dctx.stream_reader(fp, read_size=262144) as reader:
                while True:
                    chunk = reader.read(chunk_bytes)
                    if not chunk:
                        break

                    # Copy chunk data into pre-allocated buffer
                    chunk_len = len(chunk)
                    if buffer_len + chunk_len > len(buffer):
                        # Resize buffer more efficiently
                        new_size = max(len(buffer) * 2, buffer_len + chunk_len)
                        new_buffer = bytearray(new_size)
                        new_buffer[:buffer_len] = buffer_view[:buffer_len]
                        buffer = new_buffer
                        buffer_view = memoryview(buffer)

                    buffer_view[buffer_len:buffer_len + chunk_len] = chunk
                    buffer_len += chunk_len

                    n_records = buffer_len // record_size
                    if n_records == 0:
                        continue

                    valid_len = n_records * record_size

                    # Parse records directly into batch buffer
                    new_records = np.frombuffer(buffer_view[:valid_len], dtype=self.record_dtype)
                    new_records_len = len(new_records)

                    # Ensure batch_buf has enough space
                    if batch_buf_len + new_records_len > len(batch_buf):
                        # Calculate new size to accommodate all records
                        new_size = max(len(batch_buf) * 2, batch_buf_len + new_records_len)
                        new_batch_buf = np.empty((new_size,), dtype=self.record_dtype)
                        new_batch_buf[:batch_buf_len] = batch_buf[:batch_buf_len]
                        batch_buf = new_batch_buf

                    batch_buf[batch_buf_len:batch_buf_len + new_records_len] = new_records
                    batch_buf_len += new_records_len

                    # Move remaining bytes to beginning of buffer
                    remaining = buffer_len - valid_len
                    if remaining > 0:
                        buffer_view[:remaining] = buffer_view[valid_len:buffer_len]
                    buffer_len = remaining

                    while batch_buf_len >= self.batch_size:
                        yield self._process(batch_buf[:self.batch_size])
                        # More efficient shifting using memmove-like operation
                        remaining = batch_buf_len - self.batch_size
                        if remaining > 0:
                            np.copyto(batch_buf[:remaining], batch_buf[self.batch_size:batch_buf_len])
                        batch_buf_len = remaining

    def _process(self, arr: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Direct conversion without extra copies when possible
        scores = torch.from_numpy(arr["score"].astype(np.float32, copy=True))

        # In-place feature offset computation to avoid temporary arrays
        features_raw = arr["features"]
        features_adjusted = features_raw.astype(np.int64, copy=False)
        features_adjusted += self.feature_cum_offsets
        features = torch.from_numpy(features_adjusted)

        mobility = torch.from_numpy(arr["mobility"].astype(np.int64, copy=False))
        ply = torch.from_numpy(arr["ply"].astype(np.int64, copy=False))
        return scores, features, mobility, ply


    def _prefetched_feature_iterator(self, paths: List[str]):
        """Iterator with prefetching using a background thread."""
        q = queue.Queue(maxsize=self.prefetch_factor + 1)
        stop_event = threading.Event()

        def producer():
            try:
                for batch in self._feature_iterator(paths):
                    if stop_event.is_set():
                        return
                    while True:
                        try:
                            q.put_nowait(batch)
                            break
                        except queue.Full:
                            if stop_event.is_set():
                                return
                            time.sleep(0.001)
            except Exception as e:
                try:
                    q.put_nowait(e)
                except queue.Full:
                    pass
            finally:
                while True:
                    try:
                        q.put_nowait(None)
                        break
                    except queue.Full:
                        time.sleep(0.001)

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()

        try:
            while True:
                try:
                    item = q.get(timeout=0.01)
                except queue.Empty:
                    if not thread.is_alive():
                        break
                    continue

                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            stop_event.set()
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
            thread.join()


def custom_collate_fn(batch):
    assert len(batch) == 1, "Expected batch size of 1 from IterableDataset"
    return batch[0]
