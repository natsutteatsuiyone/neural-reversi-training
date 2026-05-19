import sys, os
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from sparse_linear import SparseLinear

device = torch.device("cuda")
torch.manual_seed(0)


def pooled_indices(InF, NumF, B):
    """High-contention indices: B rows drawn from a small pool of patterns."""
    pool = min(1000, InF)
    idx_pool = torch.randint(0, InF, (pool, NumF), device=device, dtype=torch.int64)
    return idx_pool[torch.randint(0, pool, (B,), device=device)].contiguous()


def reference_grad(indices, weight, grad_output):
    """grad_weight via native index_add_ (the correctness oracle)."""
    B, F = indices.shape
    OF = weight.shape[1]
    gw = torch.zeros_like(weight)
    flat_idx = indices.reshape(-1)
    flat_grad = grad_output.unsqueeze(1).expand(-1, F, -1).reshape(-1, OF)
    gw.index_add_(0, flat_idx, flat_grad)
    return gw


configs = [
    ("fast small InF=64",      64,     32,  128, 4096),
    ("fast InF=297432",        297432, 32,  128, 16384),
    ("fast OF=256",            297432, 32,  256, 8192),
    ("tiny",                   17,     5,   8,   33),
]

all_ok = True
for name, InF, NumF, OF, B in configs:
    layer = SparseLinear(InF, OF, bias=True).to(device)
    indices = pooled_indices(InF, NumF, B)
    grad_output = torch.randn(B, OF, device=device)

    out = layer(indices)
    out.backward(grad_output)
    got = layer.weight.grad.detach()

    ref = reference_grad(indices, layer.weight.detach(), grad_output)

    max_abs = (got - ref).abs().max().item()
    ref_scale = ref.abs().max().item() + 1e-9
    rel = max_abs / ref_scale
    ok = torch.allclose(got, ref, rtol=1e-3, atol=1e-3)
    all_ok &= ok
    print(f"[{'OK ' if ok else 'FAIL'}] {name:24s} max_abs={max_abs:.3e} rel={rel:.3e}")

# --- Out-of-range index safety (regression guard) ---
# An OOB feature index must NOT cause an illegal memory access; the kernels
# skip it. Valid entries must still produce correct gradients.
print()
for name, InF, NumF, OF, B in [("OOB skip fast InF=297432", 297432, 32, 384, 4096),
                               ("OOB skip fast small InF=64", 64, 32, 128, 4096)]:
    layer = SparseLinear(InF, OF, bias=True).to(device)
    indices = pooled_indices(InF, NumF, B)
    indices[7, 3] = InF * 99      # large positive OOB
    indices[11, 5] = -4           # negative OOB
    grad_output = torch.randn(B, OF, device=device)
    try:
        out = layer(indices)
        out.backward(grad_output)
        torch.cuda.synchronize()
        finite = torch.isfinite(out).all().item() and torch.isfinite(layer.weight.grad).all().item()
        ok = finite
    except RuntimeError as e:
        ok = False
        print(f"[FAIL] {name:24s} {str(e).splitlines()[0]}")
    all_ok &= ok
    if ok:
        print(f"[OK ] {name:24s} no illegal access, outputs finite")

print("\nALL CORRECT" if all_ok else "\nMISMATCH DETECTED")
sys.exit(0 if all_ok else 1)
