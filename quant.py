import torch


class FakeQuantRound(torch.autograd.Function):
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


class FakeQuantFloor(torch.autograd.Function):
    """
    Custom autograd function for fake quantization using floor operation.
    Quantizes values by flooring to the nearest lower multiple of 1/scale_factor.
    Uses straight-through estimator for the backward pass.
    """
    @staticmethod
    def forward(ctx, x, scale_factor):
        # Quantize by flooring to the nearest lower value in the discrete set
        return torch.floor(x * scale_factor) / scale_factor

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: pass the gradient through unchanged
        return grad_output, None


def q_round(x, scale_factor):
    """
    Helper function for quantization with rounding.

    Args:
        x: Input tensor to be quantized
        scale_factor: Quantization scale factor

    Returns:
        Quantized tensor using round operation
    """
    return FakeQuantRound.apply(x, scale_factor)


def q_floor(x, scale_factor):
    """
    Helper function for quantization with floor operation.

    Args:
        x: Input tensor to be quantized
        scale_factor: Quantization scale factor

    Returns:
        Quantized tensor using floor operation
    """
    return FakeQuantFloor.apply(x, scale_factor)
