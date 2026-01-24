#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Maximum number of features that can be cached in shared memory
// Set to 128 to cover typical use cases (e.g., 64 for Reversi) with some headroom
constexpr int MAX_SHARED_FEATURES = 128;

// -------------------------------------------------------------------------
// Forward Kernels
// -------------------------------------------------------------------------

// Original Fast Forward Kernel (No tiling, max 128 features)
// Optimized for NumF <= 128
template <typename scalar_t>
__global__ void onehot_sparse_linear_forward_kernel_fast(
    const int64_t* __restrict__ indices,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int num_features,
    const int out_features
) {
    __shared__ int64_t s_indices[MAX_SHARED_FEATURES];

    const int batch_idx = blockIdx.x;
    const int out_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size) return;

    // Load ALL indices (assumes num_features <= 128)
    const int64_t* batch_indices = indices + batch_idx * num_features;
    for (int i = threadIdx.x; i < num_features; i += blockDim.x) {
        s_indices[i] = batch_indices[i];
    }
    __syncthreads();

    if (out_idx >= out_features) return;

    scalar_t sum = bias ? bias[out_idx] : scalar_t(0);
    for (int f = 0; f < num_features; f++) {
        sum += weight[s_indices[f] * out_features + out_idx];
    }
    output[batch_idx * out_features + out_idx] = sum;
}

// Tiled Forward Kernel (Safe for large NumF)
template <typename scalar_t>
__global__ void onehot_sparse_linear_forward_kernel_tiled(
    const int64_t* __restrict__ indices,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int num_features,
    const int out_features
) {
    __shared__ int64_t s_indices[MAX_SHARED_FEATURES];

    const int batch_idx = blockIdx.x;
    const int out_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size) return;

    scalar_t sum = scalar_t(0);

    for (int f_start = 0; f_start < num_features; f_start += MAX_SHARED_FEATURES) {
        int f_end = (f_start + MAX_SHARED_FEATURES < num_features) ? (f_start + MAX_SHARED_FEATURES) : num_features;
        int chunk_size = f_end - f_start;

        const int64_t* batch_indices = indices + batch_idx * num_features;
        for (int i = threadIdx.x; i < chunk_size; i += blockDim.x) {
            s_indices[i] = batch_indices[f_start + i];
        }
        __syncthreads();

        if (out_idx < out_features) {
             for (int i = 0; i < chunk_size; i++) {
                 int64_t idx = s_indices[i];
                 sum += weight[idx * out_features + out_idx];
             }
        }
        __syncthreads();
    }

    if (out_idx < out_features) {
         if (bias) {
             sum += bias[out_idx];
         }
         output[batch_idx * out_features + out_idx] = sum;
    }
}

// -------------------------------------------------------------------------
// Backward Weight Kernels
// -------------------------------------------------------------------------

// Original Fast Backward Kernel (No tiling, max 128 features)
// Optimized for NumF <= 128. Returns early if out_idx >= out_features.
template <typename scalar_t>
__global__ void onehot_sparse_linear_backward_weight_kernel_fast(
    const int64_t* __restrict__ indices,
    const scalar_t* __restrict__ grad_output,
    scalar_t* __restrict__ grad_weight,
    const int batch_size,
    const int num_features,
    const int out_features
) {
    __shared__ int64_t s_indices[MAX_SHARED_FEATURES];

    const int batch_idx = blockIdx.x;
    const int out_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size) return;

    // Load ALL indices (assumes num_features <= 128)
    const int64_t* batch_indices = indices + batch_idx * num_features;
    for (int i = threadIdx.x; i < num_features; i += blockDim.x) {
        s_indices[i] = batch_indices[i];
    }
    __syncthreads();

    if (out_idx >= out_features) return;

    const scalar_t grad_out = grad_output[batch_idx * out_features + out_idx];
    for (int f = 0; f < num_features; f++) {
        atomicAdd(&grad_weight[s_indices[f] * out_features + out_idx], grad_out);
    }
}

// Tiled Backward Kernel (Safe for large NumF)
template <typename scalar_t>
__global__ void onehot_sparse_linear_backward_weight_kernel_tiled(
    const int64_t* __restrict__ indices,
    const scalar_t* __restrict__ grad_output,
    scalar_t* __restrict__ grad_weight,
    const int batch_size,
    const int num_features,
    const int out_features
) {
    __shared__ int64_t s_indices[MAX_SHARED_FEATURES];

    const int batch_idx = blockIdx.x;
    const int out_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size) return;

    for (int f_start = 0; f_start < num_features; f_start += MAX_SHARED_FEATURES) {
        int f_end = (f_start + MAX_SHARED_FEATURES < num_features) ? (f_start + MAX_SHARED_FEATURES) : num_features;
        int chunk_size = f_end - f_start;

        const int64_t* batch_indices = indices + batch_idx * num_features;
        for (int i = threadIdx.x; i < chunk_size; i += blockDim.x) {
            s_indices[i] = batch_indices[f_start + i];
        }
        __syncthreads();

        if (out_idx < out_features) {
            const scalar_t grad_out = grad_output[batch_idx * out_features + out_idx];
            for (int i = 0; i < chunk_size; i++) {
                int64_t idx = s_indices[i];
                atomicAdd(&grad_weight[idx * out_features + out_idx], grad_out);
            }
        }
        __syncthreads();
    }
}

// Dense cached kernel for small in_features (Reversi Board)
template <typename scalar_t>
__global__ void onehot_sparse_linear_backward_cached_kernel(
    const int64_t* __restrict__ indices,
    const scalar_t* __restrict__ grad_output,
    scalar_t* __restrict__ grad_weight,
    const int batch_size,
    const int num_features,
    const int in_features,
    const int out_features,
    const int batch_step
) {
    // Shared memory for gradient accumulation: [in_features, blockDim.x]
    extern __shared__ char s_buffer[];
    scalar_t* s_grad = reinterpret_cast<scalar_t*>(s_buffer);

    // s_indices allocation
    size_t s_grad_size = in_features * blockDim.x * sizeof(scalar_t);
    size_t s_grad_size_aligned = (s_grad_size + 7) / 8 * 8;
    int64_t* s_indices = reinterpret_cast<int64_t*>(s_buffer + s_grad_size_aligned);

    // 1. Initialize shared gradient to 0
    int total_elements = in_features * blockDim.x;
    for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < total_elements; i += blockDim.x * blockDim.y) {
        s_grad[i] = scalar_t(0);
    }
    __syncthreads();

    const int batch_start = blockIdx.x * batch_step;
    const int batch_end = (batch_start + batch_step < batch_size) ? (batch_start + batch_step) : batch_size;
    const int out_idx_base = blockIdx.y * blockDim.x;
    const int out_idx = out_idx_base + threadIdx.x;
    const bool valid_out = out_idx < out_features;

    // 2. Process Batches
    for (int b = batch_start; b < batch_end; b++) {
        const int64_t* batch_indices = indices + b * num_features;

        // Use tiled loading for safety, though typically NumF small here.
        for (int f_start = 0; f_start < num_features; f_start += MAX_SHARED_FEATURES) {
             int f_end = (f_start + MAX_SHARED_FEATURES < num_features) ? (f_start + MAX_SHARED_FEATURES) : num_features;
             int chunk_size = f_end - f_start;

             // Cooperatively load indices
             for (int i = threadIdx.x; i < chunk_size; i += blockDim.x) {
                 s_indices[i] = batch_indices[f_start + i];
             }
             __syncthreads();

             // Accumulate into s_grad
             if (valid_out) {
                 const scalar_t grad_out = grad_output[b * out_features + out_idx];
                 for (int i = 0; i < chunk_size; i++) {
                     int64_t idx = s_indices[i];
                     // Safety check removed for speed? No, let's keep it. InF check costs little.
                     if (idx < in_features) {
                         // Shared memory atomic
                         atomicAdd(&s_grad[idx * blockDim.x + threadIdx.x], grad_out);
                     }
                 }
             }
             __syncthreads();
        }
    }

    // 3. Write back s_grad to global grad_weight
    if (valid_out) {
        for (int idx = 0; idx < in_features; idx++) {
             scalar_t val = s_grad[idx * blockDim.x + threadIdx.x];
             if (abs(val) > 1e-9) {
                 atomicAdd(&grad_weight[idx * out_features + out_idx], val);
             }
        }
    }
}

// -------------------------------------------------------------------------
// Host Functions
// -------------------------------------------------------------------------

torch::Tensor onehot_sparse_linear_forward_cuda(
    torch::Tensor indices,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias
) {
    const int batch_size = indices.size(0);
    const int num_features = indices.size(1);
    const int out_features = weight.size(1);

    auto output = torch::empty({batch_size, out_features}, weight.options());

    // Early return for empty batch
    if (batch_size == 0) {
        return output;
    }

    const int threads = 256;
    const int blocks_y = (out_features + threads - 1) / threads;
    const dim3 blocks(batch_size, blocks_y);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        weight.scalar_type(), "onehot_sparse_linear_forward_cuda", ([&] {
        if (num_features <= MAX_SHARED_FEATURES) {
            onehot_sparse_linear_forward_kernel_fast<scalar_t><<<blocks, threads>>>(
                indices.data_ptr<int64_t>(),
                weight.data_ptr<scalar_t>(),
                bias.has_value() ? bias.value().data_ptr<scalar_t>() : nullptr,
                output.data_ptr<scalar_t>(),
                batch_size,
                num_features,
                out_features
            );
        } else {
            onehot_sparse_linear_forward_kernel_tiled<scalar_t><<<blocks, threads>>>(
                indices.data_ptr<int64_t>(),
                weight.data_ptr<scalar_t>(),
                bias.has_value() ? bias.value().data_ptr<scalar_t>() : nullptr,
                output.data_ptr<scalar_t>(),
                batch_size,
                num_features,
                out_features
            );
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }));

    return output;
}

std::tuple<torch::Tensor, torch::optional<torch::Tensor>> onehot_sparse_linear_backward_cuda(
    torch::Tensor indices,
    torch::Tensor grad_output,
    torch::Tensor weight,
    bool has_bias
) {
    const int batch_size = indices.size(0);
    const int num_features = indices.size(1);
    const int in_features = weight.size(0);
    const int out_features = weight.size(1);

    // Determine computation dtype: use fp32 for half types for numerical stability
    auto compute_dtype = weight.scalar_type();
    bool needs_cast = (compute_dtype == at::ScalarType::Half || compute_dtype == at::ScalarType::BFloat16);
    if (needs_cast) {
        compute_dtype = at::ScalarType::Float;
    }

    // Initialize grad_weight to zeros
    auto grad_weight_compute = torch::zeros({in_features, out_features},
        weight.options().dtype(compute_dtype));

    // Early return for empty batch (grad_weight is already zeros)
    if (batch_size == 0) {
        auto grad_weight = needs_cast ? grad_weight_compute.to(weight.scalar_type()) : grad_weight_compute;
        torch::optional<torch::Tensor> grad_bias = torch::nullopt;
        if (has_bias) {
            grad_bias = torch::zeros({out_features}, weight.options());
        }
        return std::make_tuple(grad_weight, grad_bias);
    }

    // Cast grad_output to compute_dtype if needed
    auto grad_output_compute = needs_cast ? grad_output.to(compute_dtype) : grad_output;

    const int threads = 256;
    const int blocks_y = (out_features + threads - 1) / threads;
    const dim3 blocks(batch_size, blocks_y);

    // Dispatch on compute_dtype (always fp32 or fp64)
    AT_DISPATCH_FLOATING_TYPES(compute_dtype, "onehot_sparse_linear_backward_cuda", ([&] {
        // 1. Try Cached Kernel (for small InFeatures e.g. Reversi Board)
        // Only use if ALL shared memory requirements fit freely.
        bool use_cached = false;
        const int max_shared_mem = 48 * 1024;
        int cached_threads = 0;
        size_t required_shm = 0;

        for (int t : {256, 128, 64, 32}) {
            size_t s_grad_size = in_features * t * sizeof(scalar_t);
            size_t s_grad_aligned = (s_grad_size + 7) / 8 * 8;
            size_t total_shm = s_grad_aligned + MAX_SHARED_FEATURES * sizeof(int64_t);

            if (total_shm <= max_shared_mem) {
                cached_threads = t;
                required_shm = total_shm;
                use_cached = true;
                break;
            }
        }

        if (use_cached) {
            const int batch_step = 32;
            const int blocks_x = (batch_size + batch_step - 1) / batch_step;
            const int blocks_y_cached = (out_features + cached_threads - 1) / cached_threads;
            const dim3 blocks_cached(blocks_x, blocks_y_cached);

            cudaFuncSetAttribute(onehot_sparse_linear_backward_cached_kernel<scalar_t>,
                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)required_shm);

            onehot_sparse_linear_backward_cached_kernel<scalar_t><<<blocks_cached, cached_threads, required_shm>>>(
                indices.data_ptr<int64_t>(),
                grad_output_compute.data_ptr<scalar_t>(),
                grad_weight_compute.data_ptr<scalar_t>(),
                batch_size,
                num_features,
                in_features,
                out_features,
                batch_step
            );
        } else if (num_features <= MAX_SHARED_FEATURES) {
            // 2. Use Fast Original Kernel (for usual large InFeatures but small NumFeatures)
            // This path has no overhead from tiling or internal barriers for non-existent tiles.
            onehot_sparse_linear_backward_weight_kernel_fast<scalar_t><<<blocks, threads>>>(
                indices.data_ptr<int64_t>(),
                grad_output_compute.data_ptr<scalar_t>(),
                grad_weight_compute.data_ptr<scalar_t>(),
                batch_size,
                num_features,
                out_features
            );
        } else {
            // 3. Fallback Tiled Kernel (for large NumFeatures)
            onehot_sparse_linear_backward_weight_kernel_tiled<scalar_t><<<blocks, threads>>>(
                indices.data_ptr<int64_t>(),
                grad_output_compute.data_ptr<scalar_t>(),
                grad_weight_compute.data_ptr<scalar_t>(),
                batch_size,
                num_features,
                out_features
            );
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }));

    // Cast back to original dtype if needed
    auto grad_weight = needs_cast ? grad_weight_compute.to(weight.scalar_type()) : grad_weight_compute;

    torch::optional<torch::Tensor> grad_bias = torch::nullopt;
    if (has_bias) {
        auto grad_bias_compute = grad_output_compute.sum(0);
        grad_bias = needs_cast ? grad_bias_compute.to(weight.scalar_type()) : grad_bias_compute;
    }

    return std::make_tuple(grad_weight, grad_bias);
}
