from random import Random
import numpy as np
import torch
import zstandard

from torch.utils.data import IterableDataset

NUM_FEATURE_PARAMS = [
    6561, 6561, 6561, 6561,
    6561, 6561, 6561, 6561,
]

NUM_FEATURES = len(NUM_FEATURE_PARAMS)
SUM_OF_FEATURES = sum(NUM_FEATURE_PARAMS)


class FeatureDataset(IterableDataset):
    def __init__(self, filepaths, batch_size, random_skipping):
        super().__init__()
        self.filepaths = filepaths
        self.batch_size = batch_size
        self.random_skipping = random_skipping
        self.rand = Random()

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
            filepaths = self.filepaths
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            filepaths = self.filepaths[worker_id::num_workers]

        return self._feature_iterator(filepaths)

    def _feature_iterator(self, filepaths):
        record_size = self.record_dtype.itemsize
        dctx = zstandard.ZstdDecompressor()
        chunk_size = record_size * self.batch_size * 10
        buffer = bytearray()
        batch_buffer = None
        skip_prob = self.random_skipping / (self.random_skipping + 1)

        for filepath in filepaths:
            with open(filepath, "rb") as fin:
                with dctx.stream_reader(fin) as reader:
                    while True:
                        chunk = reader.read(chunk_size)
                        if not chunk:
                            break
                        buffer.extend(chunk)
                        num_full_records = len(buffer) // record_size
                        if num_full_records > 0:
                            data_bytes = buffer[: num_full_records * record_size]
                            buffer = buffer[num_full_records * record_size:]
                            arr = np.frombuffer(data_bytes, dtype=self.record_dtype)
                            mask = np.random.rand(arr.shape[0]) >= skip_prob
                            selected = arr[mask]
                            if selected.size:
                                if batch_buffer is None:
                                    batch_buffer = selected
                                else:
                                    batch_buffer = np.concatenate((batch_buffer, selected))
                            while batch_buffer is not None and batch_buffer.shape[0] >= self.batch_size:
                                batch = batch_buffer[: self.batch_size]
                                yield self._process(batch)
                                batch_buffer = batch_buffer[self.batch_size:]

    def _process(self, arr, debug=False):
        feature_indices = torch.from_numpy(arr["features"] + self.feature_cum_offsets)
        mobility = torch.tensor(arr["mobility"])
        scores = torch.from_numpy(arr["score"].astype(np.float32))
        ply = torch.tensor(arr["ply"])
        if debug:
            print(feature_indices, mobility, scores, ply)
        return feature_indices, mobility, scores, ply


def custom_collate_fn(batch):
    feature_indices = torch.cat([torch.as_tensor(x[0]) for x in batch], dim=0)
    mobility = torch.cat([torch.as_tensor(x[1]) for x in batch], dim=0)
    scores = torch.cat([torch.as_tensor(x[2]) for x in batch], dim=0)
    ply = torch.cat([torch.as_tensor(x[3]) for x in batch], dim=0)
    return feature_indices, mobility, scores, ply
