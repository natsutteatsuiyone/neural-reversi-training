import sys, os
import torch
from torch.profiler import profile, ProfilerActivity

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from sparse_linear import SparseLinear

device = torch.device("cuda")
torch.manual_seed(0)

InF, NumF, OF, B = 297432, 32, 128, 16384
layer = SparseLinear(InF, OF, bias=True).to(device)
pool = 1000
idx_pool = torch.randint(0, InF, (pool, NumF), device=device, dtype=torch.int64)
indices = idx_pool[torch.randint(0, pool, (B,), device=device)].contiguous()
grad_output = torch.randn(B, OF, device=device)

out = layer(indices)
for _ in range(20):
    layer.weight.grad = None
    out.backward(grad_output, retain_graph=True)
torch.cuda.synchronize()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(50):
        layer.weight.grad = None
        out.backward(grad_output, retain_graph=True)
    torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))
