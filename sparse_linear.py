import torch
import torch.nn as nn
import torch_sparse
from typing import Optional


def sparse_linear(
    indices: torch.Tensor,
    values: torch.Tensor,
    m: int,
    n: int,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    output = torch_sparse.spmm(indices, values, m, n, weight)
    if bias is not None:
        output = output + bias
    return output


class SparseLinear(nn.Module):
    """
    Linear layer for sparse matrices using torch-sparse's spmm.

    Args:
        in_features (int): input dimension (n)
        out_features (int): output dimension
        bias (bool): whether to use bias
        init_scale (float): extra downscale factor after He init (default: 0.1)
    """

    __constants__ = ["in_features", "out_features"]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        init_scale: float = 0.25,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.init_scale = float(init_scale)

        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty((in_features, out_features), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # He (Kaiming) uniform for ReLU: a=0, nonlinearity='relu'
        nn.init.kaiming_uniform_(self.weight, a=0.0, mode="fan_in", nonlinearity="relu")

        # Extra downscale to avoid early saturation with clamp(0,1)
        with torch.no_grad():
            self.weight.mul_(self.init_scale)

        # Bias: start from 0 for stability (clamp と相性良し)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"init_scale={self.init_scale}"
        )

    def forward(
        self, indices: torch.Tensor, values: torch.Tensor, m: int, n: int
    ) -> torch.Tensor:
        return sparse_linear(indices, values, m, n, self.weight, self.bias)
