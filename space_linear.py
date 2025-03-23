import math
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

    This layer calculates the product of a sparse matrix (represented by indices and values)
    and a dense weight matrix, adding bias if needed.

    Attributes:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        weight (torch.Tensor): Weight parameter, shape is (in_features, out_features).
        bias (Optional[torch.Tensor]): Bias parameter, shape is (out_features).
    """

    __constants__ = ["in_features", "out_features"]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            bias (bool): Whether to include a bias term.
            device (Optional[torch.device]): Device to place parameters on.
            dtype (Optional[torch.dtype]): Data type of parameters.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
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
        """
        Initializes weights using Kaiming Uniform initialization,
        and biases using uniform(-bound, bound).
        The bound is calculated based on the fan_in of the weight.
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )

    def forward(
        self, indices: torch.Tensor, values: torch.Tensor, m: int, n: int
    ) -> torch.Tensor:
        """
        Computes the product of sparse matrix and weight matrix.

        Args:
            indices (torch.Tensor): Tensor of shape [2, nnz]. Indices of non-zero elements (dtype is torch.long).
            values (torch.Tensor): Tensor of shape [nnz]. Values corresponding to each index.
            m (int): Number of rows in the sparse matrix.
            n (int): Number of columns in the sparse matrix, should match in_features.

        Returns:
            torch.Tensor: Output after linear transformation. Shape is [m, out_features].
        """
        return sparse_linear(indices, values, m, n, self.weight, self.bias)
