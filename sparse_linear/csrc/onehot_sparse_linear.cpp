#include <torch/extension.h>

torch::Tensor onehot_sparse_linear_forward_cuda(
    torch::Tensor indices,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias
);

std::tuple<torch::Tensor, torch::optional<torch::Tensor>> onehot_sparse_linear_backward_cuda(
    torch::Tensor indices,
    torch::Tensor grad_output,
    torch::Tensor weight,
    bool has_bias
);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor onehot_sparse_linear_forward(
    torch::Tensor indices,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias
) {
    CHECK_INPUT(indices);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
        TORCH_CHECK(bias.value().dim() == 1, "bias must be 1D, got ", bias.value().dim(), "D");
        TORCH_CHECK(bias.value().size(0) == weight.size(1),
            "bias size (", bias.value().size(0), ") must match out_features (", weight.size(1), ")");
    }

    TORCH_CHECK(indices.dim() == 2, "indices must be 2D, got ", indices.dim(), "D");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D, got ", weight.dim(), "D");
    TORCH_CHECK(indices.scalar_type() == at::kLong, "indices must be int64, got ", indices.scalar_type());

    return onehot_sparse_linear_forward_cuda(indices, weight, bias);
}

std::tuple<torch::Tensor, torch::optional<torch::Tensor>> onehot_sparse_linear_backward(
    torch::Tensor indices,
    torch::Tensor grad_output,
    torch::Tensor weight,
    bool has_bias
) {
    CHECK_INPUT(indices);
    CHECK_INPUT(grad_output);
    CHECK_INPUT(weight);

    TORCH_CHECK(indices.dim() == 2, "indices must be 2D, got ", indices.dim(), "D");
    TORCH_CHECK(grad_output.dim() == 2, "grad_output must be 2D, got ", grad_output.dim(), "D");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D, got ", weight.dim(), "D");
    TORCH_CHECK(indices.size(0) == grad_output.size(0),
        "batch size mismatch: indices has ", indices.size(0), " but grad_output has ", grad_output.size(0));
    TORCH_CHECK(grad_output.size(1) == weight.size(1),
        "out_features mismatch: grad_output has ", grad_output.size(1), " but weight has ", weight.size(1));

    return onehot_sparse_linear_backward_cuda(indices, grad_output, weight, has_bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &onehot_sparse_linear_forward, "One-hot sparse linear forward (CUDA)");
    m.def("backward", &onehot_sparse_linear_backward, "One-hot sparse linear backward (CUDA)");
}
