import torch
import torch.nn as nn
import torch.nn.functional as F
from space_linear import SparseLinear, sparse_linear


class FakeQuantizeRound(torch.autograd.Function):
    """
    Custom autograd function for fake quantization using rounding.
    Quantizes values by rounding to the nearest multiple of 1/scale_factor.
    Uses straight-through estimator for the backward pass.
    """

    @staticmethod
    def forward(ctx, x, scale_factor):
        # Quantize by rounding to the nearest value in the discrete set
        x = torch.round(x * scale_factor) / scale_factor
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: pass the gradient through unchanged
        return grad_output, None


class FakeQuantizeFloor(torch.autograd.Function):
    """
    Custom autograd function for fake quantization using floor operation.
    Quantizes values by flooring to the nearest lower multiple of 1/scale_factor.
    Uses straight-through estimator for the backward pass.
    """

    @staticmethod
    def forward(ctx, x, scale_factor, qmin=None, qmax=None):
        # Quantize by flooring to the nearest lower value in the discrete set
        if qmin is not None and qmax is not None:
            x = torch.clamp(x, qmin, qmax)
        x = torch.floor(x * scale_factor) / scale_factor
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: pass the gradient through unchanged
        return grad_output, None, None, None


def fq_round(x, scale_factor):
    """
    Helper function for quantization with rounding.

    Args:
        x: Input tensor to be quantized
        scale_factor: Quantization scale factor

    Returns:
        Quantized tensor using round operation
    """
    return FakeQuantizeRound.apply(x, scale_factor)


def fq_floor(x, scale_factor, qmin=None, qmax=None):
    """
    Helper function for quantization with floor operation.

    Args:
        x: Input tensor to be quantized
        scale_factor: Quantization scale factor

    Returns:
        Quantized tensor using floor operation
    """
    return FakeQuantizeFloor.apply(x, scale_factor, qmin, qmax)


class FakeQuantizeLinear(nn.Linear):
    """
    Linear layer with fake quantization of weights and biases.
    Wraps nn.Linear and applies q_round to weights and biases during forward pass.
    """

    def __init__(
        self,
        in_features,
        out_features,
        weight_scale,
        bias_scale,
        bias=True,
        device=None,
        dtype=None,
    ):
        super(FakeQuantizeLinear, self).__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
        self.register_buffer("weight_scale", torch.tensor(weight_scale))
        self.register_buffer("bias_scale", torch.tensor(bias_scale))

    def forward(self, input):
        q_weight = fq_round(self.weight, self.weight_scale)
        q_bias = None
        if self.bias is not None:
            q_bias = fq_round(self.bias, self.bias_scale)
        return F.linear(input, q_weight, q_bias)


class FakeQuantizeSparseLinear(SparseLinear):
    """
    Sparse linear layer with fake quantization of weights and biases.
    Wraps SparseLinear and applies q_round to weights and biases during forward pass.
    """

    def __init__(
        self,
        in_features,
        out_features,
        weight_scale,
        bias_scale,
        bias=True,
        device=None,
        dtype=None,
    ):
        super(FakeQuantizeSparseLinear, self).__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
        self.register_buffer("weight_scale", torch.tensor(weight_scale))
        self.register_buffer("bias_scale", torch.tensor(bias_scale))

    def forward(self, indices, values, m, n):
        q_weight = fq_round(self.weight, self.weight_scale)
        q_bias = None
        if self.bias is not None:
            q_bias = fq_round(self.bias, self.bias_scale)
        return sparse_linear(indices, values, m, n, q_weight, q_bias)
