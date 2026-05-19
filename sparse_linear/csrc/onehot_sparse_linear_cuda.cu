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
// Helper Functions (compile-time foldable / forceinline: zero runtime cost)
// -------------------------------------------------------------------------

// Ceil(a / b) for non-negative integers.
__host__ __device__ __forceinline__ int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

// Smallest multiple of m that is >= a.
__host__ __device__ __forceinline__ int round_up(int a, int m) {
    return ceil_div(a, m) * m;
}

// Smallest multiple of a that is >= x.
__host__ __device__ __forceinline__ size_t align_up(size_t x, size_t a) {
    return (x + a - 1) / a * a;
}

// Branchless [0, n) check. The unsigned cast also rejects negatives; the
// template keeps the compare at idx's width (32-bit for int, 64-bit for
// int64_t) so the int path isn't needlessly widened.
template <typename T>
__host__ __device__ __forceinline__ bool in_bounds(T idx, int n) {
    using U = std::make_unsigned_t<T>;
    return static_cast<U>(idx) < static_cast<U>(n);
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
    const int in_features,
    const int out_features
) {
    // Each thread handles 4 consecutive output elements
    const int batch_idx = blockIdx.x;
    const int out_base = (blockIdx.y * blockDim.x + threadIdx.x) * 4;

    if (batch_idx >= batch_size) return;

    // Keep out-of-range threads alive for __shfl_sync, but skip their work
    const bool valid = (out_base < out_features);

    // Warp-level index sharing via shuffle
    const int lane_id = threadIdx.x & (WARP_SIZE - 1);

    const int64_t* batch_indices = indices + batch_idx * num_features;

    // Check if we can use float4 vector loads (float type + 16-byte aligned pointers)
    constexpr bool is_float = std::is_same_v<scalar_t, float>;
    const bool aligned4 = (out_features % 4 == 0)
        && (reinterpret_cast<uintptr_t>(weight) % sizeof(float4) == 0)
        && (reinterpret_cast<uintptr_t>(output) % sizeof(float4) == 0)
        && (!bias || reinterpret_cast<uintptr_t>(bias) % sizeof(float4) == 0);

    // Initialize sum with bias
    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (valid && bias && out_base + 3 < out_features) {
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
    } else if (valid && bias) {
        // Handle edge case: out_base is valid but out_base+3 exceeds out_features
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

            // Skip out-of-range threads and OOB feature indices.
            if (valid && in_bounds(idx, in_features)) {
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
    }

    // Write output
    if (valid) {
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
}

// -------------------------------------------------------------------------
// Backward Weight Kernels
// -------------------------------------------------------------------------

// Fast backward kernel with shared memory indices (for num_features <= 128)
template <typename scalar_t>
__global__ void onehot_sparse_linear_backward_weight_kernel_fast(
    const int64_t* __restrict__ indices,
    const scalar_t* __restrict__ grad_output,
    scalar_t* __restrict__ grad_weight,
    const int batch_size,
    const int num_features,
    const int in_features,
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

    // Accumulate in fp32 (scalar_t is always fp32 here; see dispatch).
    const float grad_out = static_cast<float>(__ldg(&grad_output[batch_idx * out_features + out_idx]));
    for (int f = 0; f < num_features; f++) {
        const int64_t idx = s_indices[f];
        // Defensive: callers must pass valid indices; an OOB one would
        // otherwise atomicAdd to out-of-bounds memory.
        if (in_bounds(idx, in_features)) {
            atomicAdd(&grad_weight[idx * out_features + out_idx], static_cast<scalar_t>(grad_out));
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
    const c10::cuda::OptionalCUDAGuard device_guard(weight.device());

    const int batch_size = indices.size(0);
    const int num_features = indices.size(1);
    const int in_features = weight.size(0);
    const int out_features = weight.size(1);

    auto output = torch::empty({batch_size, out_features}, weight.options());

    // Early return for empty batch
    if (batch_size == 0) {
        return output;
    }

    // fp32 / bf16 / fp16 only (the dtypes this project uses). The vectorized
    // kernel handles any num_features and out_features >= 1.
    AT_DISPATCH_V2(
        weight.scalar_type(), "onehot_sparse_linear_forward_cuda", AT_WRAP([&] {

        auto stream = at::cuda::getCurrentCUDAStream();

        // Each thread handles 4 output elements.
        const int threads = 64;
        const int blocks_y = ceil_div(out_features, threads * 4);
        const dim3 blocks(batch_size, blocks_y);

        onehot_sparse_linear_forward_kernel_vectorized<scalar_t><<<blocks, threads, 0, stream>>>(
            indices.data_ptr<int64_t>(),
            weight.data_ptr<scalar_t>(),
            bias.has_value() ? bias.value().data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            batch_size,
            num_features,
            in_features,
            out_features
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }), at::kFloat, at::kHalf, at::kBFloat16);

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

    // Zero-init grad_weight. This >100 MB buffer is HBM-write-bandwidth-bound,
    // so memset vs. torch::zeros makes no measurable difference.
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

    // The fast kernel caches all indices in shared memory; num_features is
    // fixed at 32 by the feature format, well under this bound.
    TORCH_CHECK(num_features <= MAX_SHARED_FEATURES,
        "sparse_linear backward: num_features (", num_features,
        ") must be <= ", MAX_SHARED_FEATURES);

    // compute_dtype is always fp32 here (bf16/fp16 weights promoted above).
    AT_DISPATCH_V2(compute_dtype, "onehot_sparse_linear_backward_cuda", AT_WRAP([&] {
        auto stream = at::cuda::getCurrentCUDAStream();

        // Size the block to out_features (warp-rounded, capped at 256) to
        // avoid launching idle warps.
        const int threads = min(256, round_up(out_features, WARP_SIZE));
        const int blocks_y = ceil_div(out_features, threads);
        const dim3 blocks(batch_size, blocks_y);

        onehot_sparse_linear_backward_weight_kernel_fast<scalar_t><<<blocks, threads, 0, stream>>>(
            indices.data_ptr<int64_t>(),
            grad_output_compute.data_ptr<scalar_t>(),
            grad_weight_compute.data_ptr<scalar_t>(),
            batch_size,
            num_features,
            in_features,
            out_features
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }), at::kFloat);

    // Cast back to original dtype if needed
    auto grad_weight = needs_cast ? grad_weight_compute.to(weight.scalar_type()) : grad_weight_compute;

    torch::optional<torch::Tensor> grad_bias = torch::nullopt;
    if (has_bias) {
        auto grad_bias_compute = grad_output_compute.sum(0);
        grad_bias = needs_cast ? grad_bias_compute.to(weight.scalar_type()) : grad_bias_compute;
    }

    return std::make_tuple(grad_weight, grad_bias);
}
