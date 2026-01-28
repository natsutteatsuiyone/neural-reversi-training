#!/usr/bin/env python
"""Test script for verifying C++ FeatureDataset against Python implementation."""

import struct
import sys
import tempfile
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import zstandard  # noqa: E402

# Import both implementations
from training.dataset import FeatureDataset as PyFeatureDataset  # noqa: E402
from feature_dataset import FeatureDataset as CppFeatureDataset  # noqa: E402

# Feature configuration from model_common.py
NUM_FEATURE_PARAMS = [
    6561,
    6561,
    6561,
    6561,
    6561,
    6561,
    6561,
    6561,
    6561,
    6561,
    6561,
    6561,
    6561,
    6561,
    6561,
    6561,
    6561,
    6561,
    6561,
    6561,
    19683,
    19683,
    19683,
    19683,
]
NUM_FEATURES = len(NUM_FEATURE_PARAMS)


def generate_test_data(filepath: str, num_records: int = 1000) -> None:
    """Generate synthetic test data in the expected format."""
    rng = np.random.default_rng(42)

    # Create records
    records = []
    for _ in range(num_records):
        score = np.float32(rng.uniform(-1.0, 1.0))
        features = np.array(
            [rng.integers(0, params) for params in NUM_FEATURE_PARAMS],
            dtype=np.uint16,
        )
        mobility = rng.integers(0, 64, dtype=np.uint8)
        ply = rng.integers(0, 60, dtype=np.uint8)

        # Pack record: float32 + 24*uint16 + 2*uint8 = 4 + 48 + 2 = 54 bytes
        record = struct.pack("<f", score)
        record += features.tobytes()
        record += struct.pack("<BB", mobility, ply)
        records.append(record)

    # Compress and write
    data = b"".join(records)
    cctx = zstandard.ZstdCompressor()
    compressed = cctx.compress(data)

    with open(filepath, "wb") as f:
        f.write(compressed)


def test_output_consistency():
    """Test that C++ and Python implementations produce identical output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate test files
        test_files = []
        for i in range(2):
            filepath = str(Path(tmpdir) / f"test_{i}.zst")
            generate_test_data(filepath, num_records=1024)
            test_files.append(filepath)

        batch_size = 256

        # Create both datasets
        py_dataset = PyFeatureDataset(
            filepaths=test_files,
            batch_size=batch_size,
            file_usage_ratio=1.0,
            num_features=NUM_FEATURES,
            num_feature_params=NUM_FEATURE_PARAMS,
            shuffle=False,
            prefetch_factor=0,  # Disable pipelining for deterministic comparison
        )

        cpp_dataset = CppFeatureDataset(
            filepaths=test_files,
            batch_size=batch_size,
            file_usage_ratio=1.0,
            num_features=NUM_FEATURES,
            num_feature_params=NUM_FEATURE_PARAMS,
            shuffle=False,
        )

        # Compare outputs
        py_iter = iter(py_dataset)
        cpp_iter = iter(cpp_dataset)

        batch_count = 0
        for py_batch, cpp_batch in zip(py_iter, cpp_iter):
            py_scores, py_features, py_mobility, py_ply = py_batch
            cpp_scores, cpp_features, cpp_mobility, cpp_ply = cpp_batch

            # Check shapes
            assert py_scores.shape == cpp_scores.shape, (
                f"Score shape mismatch: {py_scores.shape} vs {cpp_scores.shape}"
            )
            assert py_features.shape == cpp_features.shape, (
                f"Features shape mismatch: {py_features.shape} vs {cpp_features.shape}"
            )
            assert py_mobility.shape == cpp_mobility.shape, (
                f"Mobility shape mismatch: {py_mobility.shape} vs {cpp_mobility.shape}"
            )
            assert py_ply.shape == cpp_ply.shape, (
                f"Ply shape mismatch: {py_ply.shape} vs {cpp_ply.shape}"
            )

            # Check values
            torch.testing.assert_close(py_scores, cpp_scores, msg="Scores mismatch")
            torch.testing.assert_close(
                py_features, cpp_features, msg="Features mismatch"
            )
            torch.testing.assert_close(
                py_mobility, cpp_mobility, msg="Mobility mismatch"
            )
            torch.testing.assert_close(py_ply, cpp_ply, msg="Ply mismatch")

            batch_count += 1

        print(f"Verified {batch_count} batches - all outputs match!")


def test_throughput():
    """Measure throughput of both implementations."""
    import time

    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate larger test file
        filepath = str(Path(tmpdir) / "benchmark.zst")
        num_records = 65536 * 10  # ~10 batches worth
        generate_test_data(filepath, num_records=num_records)

        batch_size = 65536

        # Benchmark Python implementation
        py_dataset = PyFeatureDataset(
            filepaths=[filepath],
            batch_size=batch_size,
            file_usage_ratio=1.0,
            num_features=NUM_FEATURES,
            num_feature_params=NUM_FEATURE_PARAMS,
            shuffle=False,
            prefetch_factor=0,
        )

        start = time.perf_counter()
        py_batches = 0
        for _ in py_dataset:
            py_batches += 1
        py_time = time.perf_counter() - start
        py_throughput = (py_batches * batch_size) / py_time

        # Benchmark C++ implementation
        cpp_dataset = CppFeatureDataset(
            filepaths=[filepath],
            batch_size=batch_size,
            file_usage_ratio=1.0,
            num_features=NUM_FEATURES,
            num_feature_params=NUM_FEATURE_PARAMS,
            shuffle=False,
        )

        start = time.perf_counter()
        cpp_batches = 0
        for _ in cpp_dataset:
            cpp_batches += 1
        cpp_time = time.perf_counter() - start
        cpp_throughput = (cpp_batches * batch_size) / cpp_time

        print(f"Python: {py_throughput:,.0f} records/sec ({py_batches} batches)")
        print(f"C++:    {cpp_throughput:,.0f} records/sec ({cpp_batches} batches)")
        print(f"Speedup: {cpp_throughput / py_throughput:.2f}x")


if __name__ == "__main__":
    print("Testing output consistency...")
    test_output_consistency()
    print()
    print("Testing throughput...")
    test_throughput()
