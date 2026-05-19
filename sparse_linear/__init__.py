import torch
from torch import nn
from typing import Optional

from . import _C


# Register backward op first (needed by forward's autograd registration)
@torch.library.custom_op("sparse_linear::backward", mutates_args=())
def _backward_op(indices: torch.Tensor, grad_output: torch.Tensor, weight: torch.Tensor, has_bias: bool) -> list[torch.Tensor]:
    grad_weight, grad_bias = _C.backward(indices, grad_output, weight, has_bias)
    # torch.compile workaround: torch.compile requires consistent return types across all
    # code paths. We return a zero tensor instead of None when bias is not used, which is
    # converted back to None in _backward() to provide the correct gradient semantics.
    if grad_bias is None:
        grad_bias = torch.zeros(weight.size(1), dtype=weight.dtype, device=weight.device)
    return [grad_weight, grad_bias]


@_backward_op.register_fake
def _(indices: torch.Tensor, grad_output: torch.Tensor, weight: torch.Tensor, has_bias: bool) -> list[torch.Tensor]:
    grad_weight = torch.empty_like(weight)
    grad_bias = torch.empty(weight.size(1), dtype=weight.dtype, device=weight.device)
    return [grad_weight, grad_bias]


# Register forward op with torch.library for torch.compile compatibility
@torch.library.custom_op("sparse_linear::forward", mutates_args=())
def onehot_sparse_linear(indices: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
    return _C.forward(indices, weight, bias)


@onehot_sparse_linear.register_fake
def _(indices: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
    batch_size = indices.size(0)
    out_features = weight.size(1)
    return torch.empty((batch_size, out_features), dtype=weight.dtype, device=weight.device)


def _backward(ctx, grad_output: torch.Tensor):
    indices, weight = ctx.saved_tensors
    grad_output = grad_output.contiguous()
    result = _backward_op(indices, grad_output, weight, ctx.has_bias)
    grad_weight, grad_bias = result[0], result[1]
    if not ctx.has_bias:
        grad_bias = None
    return None, grad_weight, grad_bias


def _setup_context(ctx, inputs, output):
    indices, weight, bias = inputs
    # Only save tensors needed for backward; store has_bias as attribute to avoid
    # saving the bias tensor unnecessarily (its values aren't used in backward)
    ctx.save_for_backward(indices, weight)
    ctx.has_bias = bias is not None


onehot_sparse_linear.register_autograd(_backward, setup_context=_setup_context)


class SparseLinear(nn.Module):
    """
    One-hot sparse linear layer using custom CUDA kernel.

    This layer performs an efficient sparse linear operation where the input is
    a tensor of feature indices (one-hot encoded implicitly). It is optimized for
    cases like Reversi board representations where each input position maps to
    exactly one feature index.

    Constraints:
        - **CUDA-only**: requires CUDA tensors; CPU is not supported.
        - **int64 indices**: indices must be torch.int64 (torch.long), shape
          [batch_size, num_features].
        - **dtype**: weight must be float32, bfloat16, or float16 (float64 is
          not supported). bf16/fp16 backward is computed in fp32.
        - **num_features <= 128**: the backward kernel caches each sample's
          indices in shared memory; a larger second dim raises an error.
        - **Valid indices**: index values should be in [0, in_features).
          Out-of-range or negative values are defensively skipped (no crash
          or memory corruption) but produce an incorrect result for that
          entry — an out-of-range index signals an upstream data bug.
        - **Compilation note**: with the --use_fast_math NVCC flag,
          floating-point operations may have reduced precision.

    Args:
        in_features (int): input dimension (total number of possible feature indices)
        out_features (int): output dimension
        bias (bool): whether to use bias
        init_scale (float): extra downscale factor after He init (default: 0.25)
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
        nn.init.kaiming_uniform_(self.weight, a=0.0, mode="fan_in", nonlinearity="relu")
        with torch.no_grad():
            self.weight.mul_(self.init_scale)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"init_scale={self.init_scale}"
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            indices: [batch_size, num_features] tensor of feature indices (int64, CUDA).
                     All values must be in range [0, in_features).

        Returns:
            [batch_size, out_features] output tensor
        """
        return onehot_sparse_linear(indices, self.weight, self.bias)
