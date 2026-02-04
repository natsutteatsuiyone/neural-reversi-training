#include <torch/extension.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

// Maximum number of features that can be cached in shared memory
// Set to 128 to cover typical use cases (e.g., 64 for Reversi) with some headroom
constexpr int MAX_SHARED_FEATURES = 128;

// Warp size constant
constexpr int WARP_SIZE = 32;

// -------------------------------------------------------------------------
// Helper Functions
// -------------------------------------------------------------------------

// Warp-level reduction using shuffle instructions
template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// -------------------------------------------------------------------------
// Forward Kernels
// -------------------------------------------------------------------------

// Vectorized Forward Kernel using float4 for coalesced memory access
// Optimized for out_features aligned to 4
template <typename scalar_t>
__global__ void onehot_sparse_linear_forward_kernel_vectorized(
    const int64_t* __restrict__ indices,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int num_features,
    const int out_features
) {
    // Each thread handles 4 consecutive output elements
    const int batch_idx = blockIdx.x;
    const int out_base = (blockIdx.y * blockDim.x + threadIdx.x) * 4;

    if (batch_idx >= batch_size || out_base >= out_features) return;

    // Warp-level index sharing via shuffle
    const int lane_id = threadIdx.x & (WARP_SIZE - 1);

    const int64_t* batch_indices = indices + batch_idx * num_features;

    // Check if we can use float4 vector loads (float type + aligned out_features)
    constexpr bool is_float = std::is_same_v<scalar_t, float>;
    const bool aligned4 = (out_features % 4 == 0);

    // Initialize sum with bias
    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (bias && out_base + 3 < out_features) {
        if constexpr (is_float) {
            if (aligned4) {
                float4 b4 = __ldg(reinterpret_cast<const float4*>(bias + out_base));
                sum = b4;
            } else {
                sum.x = __ldg(&bias[out_base]);
                sum.y = __ldg(&bias[out_base + 1]);
                sum.z = __ldg(&bias[out_base + 2]);
                sum.w = __ldg(&bias[out_base + 3]);
            }
        } else {
            sum.x = static_cast<float>(__ldg(&bias[out_base]));
            sum.y = static_cast<float>(__ldg(&bias[out_base + 1]));
            sum.z = static_cast<float>(__ldg(&bias[out_base + 2]));
            sum.w = static_cast<float>(__ldg(&bias[out_base + 3]));
        }
    } else if (bias) {
        // Handle edge case
        if (out_base < out_features) sum.x = static_cast<float>(__ldg(&bias[out_base]));
        if (out_base + 1 < out_features) sum.y = static_cast<float>(__ldg(&bias[out_base + 1]));
        if (out_base + 2 < out_features) sum.z = static_cast<float>(__ldg(&bias[out_base + 2]));
        if (out_base + 3 < out_features) sum.w = static_cast<float>(__ldg(&bias[out_base + 3]));
    }

    // Process features using warp shuffle for index broadcast
    for (int f_base = 0; f_base < num_features; f_base += WARP_SIZE) {
        // Each lane loads one index
        int64_t my_idx = 0;
        int f_local = f_base + lane_id;
        if (f_local < num_features) {
            my_idx = __ldg(&batch_indices[f_local]);
        }

        // Process indices from all lanes in warp
        int f_end = min(f_base + WARP_SIZE, num_features);
        for (int f = f_base; f < f_end; f++) {
            // Broadcast index from lane (f - f_base) to all lanes
            int64_t idx = __shfl_sync(0xffffffff, my_idx, f - f_base);

            // Accumulate weight values
            const scalar_t* w_ptr = weight + idx * out_features + out_base;
            if (out_base + 3 < out_features) {
                if constexpr (is_float) {
                    if (aligned4) {
                        float4 w4 = __ldg(reinterpret_cast<const float4*>(w_ptr));
                        sum.x += w4.x;
                        sum.y += w4.y;
                        sum.z += w4.z;
                        sum.w += w4.w;
                    } else {
                        sum.x += __ldg(&w_ptr[0]);
                        sum.y += __ldg(&w_ptr[1]);
                        sum.z += __ldg(&w_ptr[2]);
                        sum.w += __ldg(&w_ptr[3]);
                    }
                } else {
                    sum.x += static_cast<float>(__ldg(&w_ptr[0]));
                    sum.y += static_cast<float>(__ldg(&w_ptr[1]));
                    sum.z += static_cast<float>(__ldg(&w_ptr[2]));
                    sum.w += static_cast<float>(__ldg(&w_ptr[3]));
                }
            } else {
                if (out_base < out_features) sum.x += static_cast<float>(__ldg(&w_ptr[0]));
                if (out_base + 1 < out_features) sum.y += static_cast<float>(__ldg(&w_ptr[1]));
                if (out_base + 2 < out_features) sum.z += static_cast<float>(__ldg(&w_ptr[2]));
                if (out_base + 3 < out_features) sum.w += static_cast<float>(__ldg(&w_ptr[3]));
            }
        }
    }

    // Write output
    scalar_t* out_ptr = output + batch_idx * out_features + out_base;
    if (out_base + 3 < out_features) {
        if constexpr (is_float) {
            if (aligned4) {
                *reinterpret_cast<float4*>(out_ptr) = sum;
            } else {
                out_ptr[0] = sum.x;
                out_ptr[1] = sum.y;
                out_ptr[2] = sum.z;
                out_ptr[3] = sum.w;
            }
        } else {
            out_ptr[0] = static_cast<scalar_t>(sum.x);
            out_ptr[1] = static_cast<scalar_t>(sum.y);
            out_ptr[2] = static_cast<scalar_t>(sum.z);
            out_ptr[3] = static_cast<scalar_t>(sum.w);
        }
    } else {
        if (out_base < out_features) out_ptr[0] = static_cast<scalar_t>(sum.x);
        if (out_base + 1 < out_features) out_ptr[1] = static_cast<scalar_t>(sum.y);
        if (out_base + 2 < out_features) out_ptr[2] = static_cast<scalar_t>(sum.z);
        if (out_base + 3 < out_features) out_ptr[3] = static_cast<scalar_t>(sum.w);
    }
}

// Original Fast Forward Kernel (fallback for non-vectorized cases)
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
        s_indices[i] = __ldg(&batch_indices[i]);
    }
    __syncthreads();

    if (out_idx >= out_features) return;

    float sum = bias ? static_cast<float>(__ldg(&bias[out_idx])) : 0.0f;
    for (int f = 0; f < num_features; f++) {
        sum += static_cast<float>(__ldg(&weight[s_indices[f] * out_features + out_idx]));
    }
    output[batch_idx * out_features + out_idx] = static_cast<scalar_t>(sum);
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

    float sum = 0.0f;

    for (int f_start = 0; f_start < num_features; f_start += MAX_SHARED_FEATURES) {
        int f_end = min(f_start + MAX_SHARED_FEATURES, num_features);
        int chunk_size = f_end - f_start;

        const int64_t* batch_indices = indices + batch_idx * num_features;
        for (int i = threadIdx.x; i < chunk_size; i += blockDim.x) {
            s_indices[i] = __ldg(&batch_indices[f_start + i]);
        }
        __syncthreads();

        if (out_idx < out_features) {
             for (int i = 0; i < chunk_size; i++) {
                 int64_t idx = s_indices[i];
                 sum += static_cast<float>(__ldg(&weight[idx * out_features + out_idx]));
             }
        }
        __syncthreads();
    }

    if (out_idx < out_features) {
         if (bias) {
             sum += static_cast<float>(__ldg(&bias[out_idx]));
         }
         output[batch_idx * out_features + out_idx] = static_cast<scalar_t>(sum);
    }
}

// -------------------------------------------------------------------------
// Backward Weight Kernels with Warp-level Reduction
// -------------------------------------------------------------------------

// Warp-optimized backward kernel that reduces atomic contention
// Uses warp-level primitives to aggregate gradients before atomic operations
template <typename scalar_t>
__global__ void onehot_sparse_linear_backward_weight_kernel_warp_optimized(
    const int64_t* __restrict__ indices,
    const scalar_t* __restrict__ grad_output,
    scalar_t* __restrict__ grad_weight,
    const int batch_size,
    const int num_features,
    const int out_features
) {
    // Process multiple batches per block for better occupancy
    const int batches_per_block = 4;
    const int batch_base = blockIdx.x * batches_per_block;
    const int out_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (out_idx >= out_features) return;

    const int lane_id = threadIdx.x & (WARP_SIZE - 1);

    // Process each batch
    for (int b_offset = 0; b_offset < batches_per_block; b_offset++) {
        int batch_idx = batch_base + b_offset;
        if (batch_idx >= batch_size) break;

        const int64_t* batch_indices = indices + batch_idx * num_features;
        const float grad_out = static_cast<float>(__ldg(&grad_output[batch_idx * out_features + out_idx]));

        // Process features with warp shuffle for index broadcast
        for (int f_base = 0; f_base < num_features; f_base += WARP_SIZE) {
            // Each lane loads one index
            int64_t my_idx = -1;
            int f_local = f_base + lane_id;
            if (f_local < num_features) {
                my_idx = __ldg(&batch_indices[f_local]);
            }

            // Process indices from all lanes in warp
            int f_count = min(WARP_SIZE, num_features - f_base);
            for (int f = 0; f < f_count; f++) {
                // Broadcast index from lane f to all lanes
                int64_t idx = __shfl_sync(0xffffffff, my_idx, f);
                if (idx >= 0) {
                    atomicAdd(&grad_weight[idx * out_features + out_idx], static_cast<scalar_t>(grad_out));
                }
            }
        }
    }
}

// Fast backward kernel with warp shuffle (for num_features <= 128)
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
        s_indices[i] = __ldg(&batch_indices[i]);
    }
    __syncthreads();

    if (out_idx >= out_features) return;

    const float grad_out = static_cast<float>(__ldg(&grad_output[batch_idx * out_features + out_idx]));
    for (int f = 0; f < num_features; f++) {
        atomicAdd(&grad_weight[s_indices[f] * out_features + out_idx], static_cast<scalar_t>(grad_out));
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
        int f_end = min(f_start + MAX_SHARED_FEATURES, num_features);
        int chunk_size = f_end - f_start;

        const int64_t* batch_indices = indices + batch_idx * num_features;
        for (int i = threadIdx.x; i < chunk_size; i += blockDim.x) {
            s_indices[i] = __ldg(&batch_indices[f_start + i]);
        }
        __syncthreads();

        if (out_idx < out_features) {
            const float grad_out = static_cast<float>(__ldg(&grad_output[batch_idx * out_features + out_idx]));
            for (int i = 0; i < chunk_size; i++) {
                int64_t idx = s_indices[i];
                atomicAdd(&grad_weight[idx * out_features + out_idx], static_cast<scalar_t>(grad_out));
            }
        }
        __syncthreads();
    }
}

// Dense cached kernel for small in_features (Reversi Board) with warp-level optimization
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

    // s_indices allocation - use int32 for smaller in_features to save memory
    size_t s_grad_size = in_features * blockDim.x * sizeof(scalar_t);
    size_t s_grad_size_aligned = (s_grad_size + 7) / 8 * 8;

    // Index storage - use int32 when possible to save shared memory
    int* s_indices_32 = reinterpret_cast<int*>(s_buffer + s_grad_size_aligned);
    int64_t* s_indices_64 = reinterpret_cast<int64_t*>(s_buffer + s_grad_size_aligned);
    const bool use_int32 = (in_features < 65536);

    // 1. Initialize shared gradient to 0
    int total_elements = in_features * blockDim.x;
    for (int i = threadIdx.x; i < total_elements; i += blockDim.x) {
        s_grad[i] = scalar_t(0);
    }
    __syncthreads();

    const int batch_start = blockIdx.x * batch_step;
    const int batch_end = min(batch_start + batch_step, batch_size);
    const int out_idx_base = blockIdx.y * blockDim.x;
    const int out_idx = out_idx_base + threadIdx.x;
    const bool valid_out = out_idx < out_features;

    // 2. Process Batches
    for (int b = batch_start; b < batch_end; b++) {
        const int64_t* batch_indices = indices + b * num_features;

        // Use tiled loading for safety, though typically NumF small here.
        for (int f_start = 0; f_start < num_features; f_start += MAX_SHARED_FEATURES) {
             int f_end = min(f_start + MAX_SHARED_FEATURES, num_features);
             int chunk_size = f_end - f_start;

             // Cooperatively load indices with type optimization
             for (int i = threadIdx.x; i < chunk_size; i += blockDim.x) {
                 if (use_int32) {
                     s_indices_32[i] = static_cast<int>(__ldg(&batch_indices[f_start + i]));
                 } else {
                     s_indices_64[i] = __ldg(&batch_indices[f_start + i]);
                 }
             }
             __syncthreads();

             // Accumulate into s_grad
             if (valid_out) {
                 const float grad_out = static_cast<float>(__ldg(&grad_output[b * out_features + out_idx]));
                 for (int i = 0; i < chunk_size; i++) {
                     int idx = use_int32 ? s_indices_32[i] : static_cast<int>(s_indices_64[i]);
                     if (idx < in_features) {
                         // Shared memory atomic
                         atomicAdd(&s_grad[idx * blockDim.x + threadIdx.x], static_cast<scalar_t>(grad_out));
                     }
                 }
             }
             __syncthreads();
        }
    }

    // 3. Write back s_grad to global grad_weight using vectorized writes when possible
    if (valid_out) {
        for (int idx = 0; idx < in_features; idx++) {
             scalar_t val = s_grad[idx * blockDim.x + threadIdx.x];
             // Only write non-zero values to reduce atomic contention
             // Use a small epsilon to handle floating point comparison
             if (val != scalar_t(0)) {
                 atomicAdd(&grad_weight[idx * out_features + out_idx], val);
             }
        }
    }
}

// -------------------------------------------------------------------------
// Host Functions
// -------------------------------------------------------------------------

// Compute optimal batch step based on shared memory requirements
int computeOptimalBatchStep(int in_features, int threads, size_t available_shm) {
    size_t s_grad_size = in_features * threads * sizeof(float);
    size_t s_grad_aligned = (s_grad_size + 7) / 8 * 8;
    size_t indices_size = MAX_SHARED_FEATURES * sizeof(int64_t);
    size_t total = s_grad_aligned + indices_size;
    return (total <= available_shm) ? 64 : 8;
}

torch::Tensor onehot_sparse_linear_forward_cuda(
    torch::Tensor indices,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias
) {
    const c10::cuda::OptionalCUDAGuard device_guard(weight.device());

    const int batch_size = indices.size(0);
    const int num_features = indices.size(1);
    const int out_features = weight.size(1);

    auto output = torch::empty({batch_size, out_features}, weight.options());

    // Early return for empty batch
    if (batch_size == 0) {
        return output;
    }

    AT_DISPATCH_V2(
        weight.scalar_type(), "onehot_sparse_linear_forward_cuda", AT_WRAP([&] {

        auto stream = at::cuda::getCurrentCUDAStream();

        // Use vectorized kernel when out_features >= 4 for better memory coalescing
        const bool use_vectorized = (out_features >= 4);

        if (use_vectorized) {
            // Vectorized kernel: each thread handles 4 output elements
            const int threads = 64;  // Reduced thread count since each thread does 4x work
            const int blocks_y = (out_features + threads * 4 - 1) / (threads * 4);
            const dim3 blocks(batch_size, blocks_y);

            onehot_sparse_linear_forward_kernel_vectorized<scalar_t><<<blocks, threads, 0, stream>>>(
                indices.data_ptr<int64_t>(),
                weight.data_ptr<scalar_t>(),
                bias.has_value() ? bias.value().data_ptr<scalar_t>() : nullptr,
                output.data_ptr<scalar_t>(),
                batch_size,
                num_features,
                out_features
            );
        } else if (num_features <= MAX_SHARED_FEATURES) {
            const int threads = 256;
            const int blocks_y = (out_features + threads - 1) / threads;
            const dim3 blocks(batch_size, blocks_y);

            onehot_sparse_linear_forward_kernel_fast<scalar_t><<<blocks, threads, 0, stream>>>(
                indices.data_ptr<int64_t>(),
                weight.data_ptr<scalar_t>(),
                bias.has_value() ? bias.value().data_ptr<scalar_t>() : nullptr,
                output.data_ptr<scalar_t>(),
                batch_size,
                num_features,
                out_features
            );
        } else {
            const int threads = 256;
            const int blocks_y = (out_features + threads - 1) / threads;
            const dim3 blocks(batch_size, blocks_y);

            onehot_sparse_linear_forward_kernel_tiled<scalar_t><<<blocks, threads, 0, stream>>>(
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
    }), AT_EXPAND(AT_FLOATING_TYPES), at::kHalf, at::kBFloat16);

    return output;
}

std::tuple<torch::Tensor, torch::optional<torch::Tensor>> onehot_sparse_linear_backward_cuda(
    torch::Tensor indices,
    torch::Tensor grad_output,
    torch::Tensor weight,
    bool has_bias
) {
    const c10::cuda::OptionalCUDAGuard device_guard(weight.device());

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

    // Dispatch on compute_dtype (always fp32 or fp64)
    AT_DISPATCH_V2(compute_dtype, "onehot_sparse_linear_backward_cuda", AT_WRAP([&] {
        auto stream = at::cuda::getCurrentCUDAStream();

        // 1. Try Cached Kernel (for small InFeatures e.g. Reversi Board)
        // Only use if ALL shared memory requirements fit freely.
        bool use_cached = false;
        const int max_shared_mem = 48 * 1024;
        int cached_threads = 0;
        size_t required_shm = 0;

        for (int t : {256, 128, 64, 32}) {
            size_t s_grad_size = in_features * t * sizeof(scalar_t);
            size_t s_grad_aligned = (s_grad_size + 7) / 8 * 8;
            // Account for index type optimization: use int32 when in_features < 65536
            size_t idx_size = (in_features < 65536) ?
                MAX_SHARED_FEATURES * sizeof(int) :
                MAX_SHARED_FEATURES * sizeof(int64_t);
            size_t total_shm = s_grad_aligned + idx_size;

            if (total_shm <= max_shared_mem) {
                cached_threads = t;
                required_shm = total_shm;
                use_cached = true;
                break;
            }
        }

        if (use_cached) {
            // Dynamic batch step computation
            const int batch_step = computeOptimalBatchStep(in_features, cached_threads, max_shared_mem);
            const int blocks_x = (batch_size + batch_step - 1) / batch_step;
            const int blocks_y_cached = (out_features + cached_threads - 1) / cached_threads;
            const dim3 blocks_cached(blocks_x, blocks_y_cached);

            auto cuda_err = cudaFuncSetAttribute(onehot_sparse_linear_backward_cached_kernel<scalar_t>,
                 cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(required_shm));
            if (cuda_err != cudaSuccess) {
                // Fall back to non-cached kernel path
                use_cached = false;
            }
        }

        if (use_cached) {
            // Dynamic batch step computation (recalculate since use_cached may have changed)
            const int batch_step = computeOptimalBatchStep(in_features, cached_threads, max_shared_mem);
            const int blocks_x = (batch_size + batch_step - 1) / batch_step;
            const int blocks_y_cached = (out_features + cached_threads - 1) / cached_threads;
            const dim3 blocks_cached(blocks_x, blocks_y_cached);

            onehot_sparse_linear_backward_cached_kernel<scalar_t><<<blocks_cached, cached_threads, required_shm, stream>>>(
                indices.data_ptr<int64_t>(),
                grad_output_compute.data_ptr<scalar_t>(),
                grad_weight_compute.data_ptr<scalar_t>(),
                batch_size,
                num_features,
                in_features,
                out_features,
                batch_step
            );
        } else {
            // Use warp-optimized kernel for better atomic performance
            const int threads = 256;
            const int batches_per_block = 4;
            const int blocks_x = (batch_size + batches_per_block - 1) / batches_per_block;
            const int blocks_y = (out_features + threads - 1) / threads;
            const dim3 blocks(blocks_x, blocks_y);

            if (num_features <= MAX_SHARED_FEATURES) {
                // Fast kernel with shared memory for indices
                const dim3 blocks_fast(batch_size, blocks_y);
                onehot_sparse_linear_backward_weight_kernel_fast<scalar_t><<<blocks_fast, threads, 0, stream>>>(
                    indices.data_ptr<int64_t>(),
                    grad_output_compute.data_ptr<scalar_t>(),
                    grad_weight_compute.data_ptr<scalar_t>(),
                    batch_size,
                    num_features,
                    out_features
                );
            } else {
                // Tiled kernel for large num_features
                const dim3 blocks_tiled(batch_size, blocks_y);
                onehot_sparse_linear_backward_weight_kernel_tiled<scalar_t><<<blocks_tiled, threads, 0, stream>>>(
                    indices.data_ptr<int64_t>(),
                    grad_output_compute.data_ptr<scalar_t>(),
                    grad_weight_compute.data_ptr<scalar_t>(),
                    batch_size,
                    num_features,
                    out_features
                );
            }
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }), AT_EXPAND(AT_FLOATING_TYPES));

    // Cast back to original dtype if needed
    auto grad_weight = needs_cast ? grad_weight_compute.to(weight.scalar_type()) : grad_weight_compute;

    torch::optional<torch::Tensor> grad_bias = torch::nullopt;
    if (has_bias) {
        auto grad_bias_compute = grad_output_compute.sum(0);
        grad_bias = needs_cast ? grad_bias_compute.to(weight.scalar_type()) : grad_bias_compute;
    }

    return std::make_tuple(grad_weight, grad_bias);
}
