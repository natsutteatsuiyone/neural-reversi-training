import torch
import torch.nn as nn
import time
import sys
import os

# Add parent directory to path to allow importing sparse_linear
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sparse_linear import SparseLinear

device = None # Declare device globally

def run_benchmark(name, in_features, num_features, batch_size=1024*16):
    print(f"\n=== Benchmark: {name} ===")
    out_features = 128
    print(f"Config: B={batch_size}, NumF={num_features}, InF={in_features}, OutF={out_features}")

    layer = SparseLinear(in_features, out_features, bias=True).to(device)

    # Simulate High Contention
    # Create pool of repeated indices
    pool_size = 1000
    if pool_size > in_features: pool_size = in_features

    indices_pool = torch.randint(0, in_features, (pool_size, num_features), device=device, dtype=torch.int64)
    indices = indices_pool[torch.randint(0, pool_size, (batch_size,), device=device)]

    # Verify shape
    indices = indices.view(batch_size, num_features)

    # Warmup
    for _ in range(10):
        out = layer(indices)
        loss = out.sum()
        loss.backward()
        layer.zero_grad()
    torch.cuda.synchronize()

    # Measure Forward
    start = time.time()
    for _ in range(100):
        out = layer(indices)
    torch.cuda.synchronize()
    fwd_time = (time.time() - start) / 100 * 1000
    print(f"Forward average time: {fwd_time:.3f} ms")

    # Measure Backward
    grad_output = torch.randn(batch_size, out_features, device=device)
    start = time.time()
    for _ in range(100):
        out = layer(indices)
        out.backward(grad_output)

    # Total time forward+backward
    total_time = (time.time() - start) / 100 * 1000
    bwd_time = total_time - fwd_time
    print(f"Backward average time (estimated): {bwd_time:.3f} ms")
    print(f"Total time: {total_time:.3f} ms")

    # Measure Pytorch Native Implementation (index_add_)
    print("\n--- Benchmarking Native index_add_ ---")

    # Pre-allocate for fairness (though logic typically allocates)
    grad_weight_native = torch.zeros_like(layer.weight)

    start = time.time()
    for _ in range(100):
        # Emulate backward: grad_weight.index_add_
        # We need to flatten indices and expand grad_output
        flat_indices = indices.view(-1)
        # grad_output: [B, O] -> [B, F, O] -> [B*F, O]
        flat_grad = grad_output.unsqueeze(1).expand(-1, num_features, -1).reshape(-1, out_features)

        grad_weight_native.zero_()
        grad_weight_native.index_add_(0, flat_indices, flat_grad)

    torch.cuda.synchronize()
    native_time = (time.time() - start) / 100 * 1000
    print(f"Native index_add_ time: {native_time:.3f} ms")
    print(f"Speedup vs Custom: {bwd_time / native_time:.2f}x")

    return fwd_time, bwd_time

def benchmark():
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA not available")
        return

    # Case 1: Reversi Board (Small InCh, High Contention) -> Should use Cached Kernel
    print("\n-------------------------------------------------------------")
    print("Case 1: Reversi Board (Small InF, High Contention)")
    print("Expectation: Cached Kernel (Fastest)")
    fwd_s, bwd_s = run_benchmark("Reversi Board", in_features=64, num_features=24)

    # Case 2: Pattern Features (Large InF, High Contention) -> Should use Standard Kernel
    # model_common.py: InF ~ 215k
    print("\n-------------------------------------------------------------")
    print("Case 2: Pattern Features (Large InF, High Contention)")
    print("Expectation: Standard Kernel (Robust)")
    fwd_l, bwd_l = run_benchmark("Pattern Features", in_features=209952, num_features=24)

    # Safety check: indices with num_features > 128
    print("\n-------------------------------------------------------------")
    print("Case 3: Safety Check (NumF=256)")
    try:
        in_features = 209952
        batch_size = 1024 * 16
        layer = SparseLinear(in_features, 128, bias=True).to(device)
        indices_large = torch.randint(0, in_features, (batch_size, 256), device=device, dtype=torch.int64)
        out = layer(indices_large)
        print("Success: Forward with NumF=256")
        out.sum().backward()
        print("Success: Backward with NumF=256")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    benchmark()
