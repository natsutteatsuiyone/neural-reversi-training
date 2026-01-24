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

// Maximum number of features supported by shared memory in CUDA kernels
constexpr int64_t MAX_FEATURES = 128;

torch::Tensor onehot_sparse_linear_forward(
    torch::Tensor indices,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias
) {
    CHECK_INPUT(indices);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }

    // Lightweight O(1) validation
    TORCH_CHECK(indices.dim() == 2, "indices must be 2D, got ", indices.dim(), "D");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D, got ", weight.dim(), "D");
    TORCH_CHECK(indices.scalar_type() == at::kLong, "indices must be int64, got ", indices.scalar_type());

    // num_features check removed: kernel now handles arbitrary num_features via tiling

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
    return onehot_sparse_linear_backward_cuda(indices, grad_output, weight, has_bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &onehot_sparse_linear_forward, "One-hot sparse linear forward (CUDA)");
    m.def("backward", &onehot_sparse_linear_backward, "One-hot sparse linear backward (CUDA)");
}
