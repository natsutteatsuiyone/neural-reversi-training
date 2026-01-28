import queue
import random
import sys
import threading
import warnings
from typing import Iterable, List, Tuple

import numpy as np
import torch
import zstandard

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
        decompress_workers: int = 1,
    ):
        super().__init__()

        if not (0.0 < file_usage_ratio <= 1.0):
            raise ValueError("file_fraction_per_epoch must be in the range (0.0, 1.0].")

        self.filepaths = list(filepaths)
        self.batch_size = batch_size
        self.file_usage_ratio = file_usage_ratio
        self.shuffle = shuffle
        self.prefetch_factor = prefetch_factor
        self.decompress_workers = decompress_workers

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

        self.feature_cum_offsets = np.cumsum(
            [0] + num_feature_params[:-1], dtype=np.int64
        )

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
            return self._pipelined_feature_iterator(selected_paths)
        return self._feature_iterator(selected_paths)

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

    def _process(
        self, arr: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scores = torch.from_numpy(arr["score"].copy())

        features_raw = arr["features"]
        features_adjusted = features_raw.astype(np.int64)
        features_adjusted += self.feature_cum_offsets
        features = torch.from_numpy(features_adjusted)

        mobility = torch.from_numpy(arr["mobility"].astype(np.int64, copy=False))
        ply = torch.from_numpy(arr["ply"].astype(np.int64, copy=False))
        return scores, features, mobility, ply

    def _pipelined_feature_iterator(
        self, paths: Iterable[str]
    ) -> Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Worker pool for parallel file processing with per-worker queues.

        Files are pre-distributed to workers, each writing to its own queue.
        The consumer reads from queues in round-robin order for balanced
        throughput. Sentinel values (None) signal worker completion.

        With a free-threaded Python 3.14+ build (e.g., python3.14t) and
        PYTHON_GIL=0, workers run truly in parallel. Under standard Python
        builds, GIL-releasing C extensions (zstandard, NumPy) and I/O overlap
        still provide concurrency benefits.
        """
        paths_list = list(paths)
        if not paths_list:
            return

        num_workers = min(self.decompress_workers, len(paths_list))

        # Per-worker output queues
        worker_queues: List[queue.Queue] = [
            queue.Queue(maxsize=self.prefetch_factor) for _ in range(num_workers)
        ]

        def worker(worker_paths: List[str], out_queue: queue.Queue):
            """Process assigned files: read -> decompress -> batch -> queue."""
            buffer = bytearray()
            dctx = zstandard.ZstdDecompressor()
            current_path = None

            try:
                for path in worker_paths:
                    current_path = path
                    with open(path, "rb") as fp, dctx.stream_reader(fp) as reader:
                        while True:
                            chunk = reader.read(self.chunk_bytes)
                            if not chunk:
                                break

                            buffer.extend(chunk)

                            total_records = len(buffer) // self.record_size
                            if total_records < self.batch_size:
                                continue

                            full_batches = total_records // self.batch_size
                            cutoff = full_batches * self.batch_bytes
                            records = np.frombuffer(
                                bytes(buffer[:cutoff]), dtype=self.record_dtype
                            )

                            for start in range(0, len(records), self.batch_size):
                                batch = records[start : start + self.batch_size]
                                out_queue.put(self._process(batch))

                            del buffer[:cutoff]

            except Exception as exc:
                # Wrap exception with file context and preserve traceback
                exc_info = sys.exc_info()
                try:
                    wrapped = type(exc)(f"Error processing {current_path}: {exc}")
                except TypeError:
                    wrapped = RuntimeError(f"Error processing {current_path}: {exc}")
                wrapped.__cause__ = exc
                wrapped.__traceback__ = exc_info[2]
                out_queue.put(wrapped)
            finally:
                out_queue.put(None)  # Sentinel for completion

        # Distribute files and start workers
        threads = []
        for i in range(num_workers):
            worker_paths = paths_list[i::num_workers]
            t = threading.Thread(
                target=worker,
                args=(worker_paths, worker_queues[i]),
                daemon=True,
                name=f"data_worker_{i}",
            )
            threads.append(t)
            t.start()

        # Round-robin consumption from worker queues
        active = set(range(num_workers))
        current = 0
        first_exception = None

        try:
            while active:
                if current not in active:
                    current = (current + 1) % num_workers
                    continue

                try:
                    item = worker_queues[current].get(timeout=0.1)
                except queue.Empty:
                    current = (current + 1) % num_workers
                    continue

                if item is None:
                    active.discard(current)
                elif isinstance(item, Exception):
                    # Collect all exceptions from remaining workers before raising
                    first_exception = item
                    additional_exceptions = []
                    for idx in list(active):
                        try:
                            while True:
                                remaining = worker_queues[idx].get_nowait()
                                if remaining is None:
                                    break
                                if isinstance(remaining, Exception):
                                    additional_exceptions.append(remaining)
                        except queue.Empty:
                            pass

                    if additional_exceptions:
                        warnings.warn(
                            f"{len(additional_exceptions)} additional worker exception(s) "
                            f"occurred: {[str(e) for e in additional_exceptions]}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                    raise first_exception
                else:
                    yield item

                current = (current + 1) % num_workers

        finally:
            for t in threads:
                t.join(timeout=2.0)
                if t.is_alive():
                    warnings.warn(
                        f"Worker thread {t.name} did not terminate within timeout. "
                        "This may indicate a deadlock or blocked I/O.",
                        RuntimeWarning,
                        stacklevel=2,
                    )


def custom_collate_fn(batch):
    assert len(batch) == 1, "Expected batch size of 1 from IterableDataset"
    return batch[0]
