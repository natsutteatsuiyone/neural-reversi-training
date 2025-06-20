import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from sparse_linear import SparseLinear, sparse_linear


class FakeQuantizeRound(torch.autograd.Function):
    """Fake quantization with rounding using straight-through estimator."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale_factor: float) -> torch.Tensor:
        return torch.round(x * scale_factor) / scale_factor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output, None


class FakeQuantizeFloor(torch.autograd.Function):
    """Fake quantization with floor operation using straight-through estimator."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        scale_factor: float,
        qmin: Optional[float] = None,
        qmax: Optional[float] = None,
    ) -> torch.Tensor:
        if qmin is not None and qmax is not None:
            x = torch.clamp(x, qmin, qmax)
        return torch.floor(x * scale_factor) / scale_factor

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None]:
        return grad_output, None, None, None


def fq_round(x: torch.Tensor, scale_factor: float) -> torch.Tensor:
    """Quantize tensor using rounding."""
    return FakeQuantizeRound.apply(x, scale_factor)


def fq_floor(
    x: torch.Tensor,
    scale_factor: float,
    qmin: Optional[float] = None,
    qmax: Optional[float] = None,
) -> torch.Tensor:
    """Quantize tensor using floor operation."""
    return FakeQuantizeFloor.apply(x, scale_factor, qmin, qmax)


class FakeQuantizeLinear(nn.Linear):
    """Linear layer with fake quantization of weights and biases."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_scale: float,
        bias_scale: float,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
        self.register_buffer("weight_scale", torch.tensor(weight_scale))
        self.register_buffer("bias_scale", torch.tensor(bias_scale))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        q_weight = fq_round(self.weight, self.weight_scale)
        q_bias = fq_round(self.bias, self.bias_scale) if self.bias is not None else None
        return F.linear(input, q_weight, q_bias)


class FakeQuantizeSparseLinear(SparseLinear):
    """Sparse linear layer with fake quantization of weights and biases."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_scale: float,
        bias_scale: float,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
        self.register_buffer("weight_scale", torch.tensor(weight_scale))
        self.register_buffer("bias_scale", torch.tensor(bias_scale))

    def forward(
        self, indices: torch.Tensor, values: torch.Tensor, m: int, n: int
    ) -> torch.Tensor:
        q_weight = fq_round(self.weight, self.weight_scale)
        q_bias = fq_round(self.bias, self.bias_scale) if self.bias is not None else None
        return sparse_linear(indices, values, m, n, q_weight, q_bias)
