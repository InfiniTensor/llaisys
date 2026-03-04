#include "self_attention_nvidia.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cmath>
#include <limits>

namespace llaisys::ops::nvidia {

// ============================================================================
// Type Conversion Helpers
// ============================================================================
template<typename T> __device__ inline float to_float(T v);
template<> __device__ inline float to_float<float>(float v) { return v; }
template<> __device__ inline float to_float<half>(half v) { return __half2float(v); }
template<> __device__ inline float to_float<nv_bfloat16>(nv_bfloat16 v) { return __bfloat162float(v); }

template<typename T> __device__ inline T from_float(float v);
template<> __device__ inline float from_float<float>(float v) { return v; }
template<> __device__ inline half from_float<half>(float v) { return __float2half(v); }
template<> __device__ inline nv_bfloat16 from_float<nv_bfloat16>(float v) { return __float2bfloat16(v); }

// ============================================================================
// Kernels
// ============================================================================

// Gather Q: [seq, head, dim] -> float [head, seq, dim]
template<typename T>
__global__ void gather_all_q_kernel(
    float *dst, const T *src,
    size_t seq_len, size_t n_head, size_t head_dim,
    size_t s0, size_t s1, size_t s2
) {
    size_t total = n_head * seq_len * head_dim;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    size_t d = tid % head_dim;
    size_t s = (tid / head_dim) % seq_len;
    size_t h = tid / (head_dim * seq_len);

    dst[h * seq_len * head_dim + s * head_dim + d] =
        to_float(src[s * s0 + h * s1 + d * s2]);
}

// Gather KV + GQA Expand: [total, kv_head, dim] -> float [head, total, dim]
template<typename T>
__global__ void gather_expand_kv_kernel(
    float *dst, const T *src,
    size_t total_len, size_t n_head, size_t n_kv_head, size_t dim,
    size_t group_size,
    size_t s0, size_t s1, size_t s2
) {
    size_t total = n_head * total_len * dim;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    size_t d = tid % dim;
    size_t t = (tid / dim) % total_len;
    size_t h = tid / (dim * total_len);
    size_t kv_h = h / group_size;

    dst[h * total_len * dim + t * dim + d] =
        to_float(src[t * s0 + kv_h * s1 + d * s2]);
}

// Scatter O: float [head, seq, dim] -> [seq, head, dim]
template<typename T>
__global__ void scatter_all_heads_kernel(
    T *dst, const float *src,
    size_t seq_len, size_t n_head, size_t v_dim,
    size_t s0, size_t s1, size_t s2
) {
    size_t total = n_head * seq_len * v_dim;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    size_t d = tid % v_dim;
    size_t s = (tid / v_dim) % seq_len;
    size_t h = tid / (v_dim * seq_len);

    dst[s * s0 + h * s1 + d * s2] =
        from_float<T>(src[h * seq_len * v_dim + s * v_dim + d]);
}

__global__ void scale_kernel(float *data, float scale, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) data[tid] *= scale;
}

__global__ void causal_mask_batched_kernel(
    float *scores,
    size_t n_head, size_t seq_len, size_t total_len
) {
    size_t per_head = seq_len * total_len;
    size_t total = n_head * per_head;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    size_t pos_in_head = tid % per_head;
    size_t i = pos_in_head / total_len;
    size_t j = pos_in_head % total_len;
    size_t current_pos = total_len - seq_len + i;

    if (j > current_pos) scores[tid] = -1e30f;
}

__global__ void softmax_row_kernel(float *data, size_t num_rows, size_t row_len) {
    size_t row = blockIdx.x;
    if (row >= num_rows) return;

    extern __shared__ float sdata[];
    float *row_data = data + row * row_len;

    float local_max = -1e30f;
    for (size_t j = threadIdx.x; j < row_len; j += blockDim.x) {
        float val = row_data[j];
        if (val > local_max) local_max = val;
    }
    sdata[threadIdx.x] = local_max;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && sdata[threadIdx.x + s] > sdata[threadIdx.x])
            sdata[threadIdx.x] = sdata[threadIdx.x + s];
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (size_t j = threadIdx.x; j < row_len; j += blockDim.x) {
        float val = expf(row_data[j] - max_val);
        row_data[j] = val;
        local_sum += val;
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / (sdata[0] + 1e-6f);
    __syncthreads();

    for (size_t j = threadIdx.x; j < row_len; j += blockDim.x) {
        row_data[j] *= inv_sum;
    }
}

// ============================================================================
// Utilities: cuBLAS Handle & Cached Buffers
// ============================================================================
static cublasHandle_t get_cublas_handle() {
    static thread_local cublasHandle_t handle = nullptr;
    if (!handle) {
        cublasStatus_t status = cublasCreate(&handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "cuBLAS handle creation failed: %d\n", (int)status);
        }
    }
    return handle;
}

static float *s_q_all = nullptr;
static float *s_k_all = nullptr;
static float *s_v_all = nullptr;
static float *s_scores = nullptr;
static float *s_o_all = nullptr;
static size_t s_q_all_sz = 0;
static size_t s_k_all_sz = 0;
static size_t s_v_all_sz = 0;
static size_t s_scores_sz = 0;
static size_t s_o_all_sz = 0;

static void ensure_buf(float *&ptr, size_t &cur, size_t need) {
    if (need <= cur) return;
    if (ptr) cudaFree(ptr);
    cudaError_t err = cudaMalloc(&ptr, need);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc failed: %s\n", cudaGetErrorString(err));
        ptr = nullptr;
        cur = 0;
        return;
    }
    cur = need;
}

// ============================================================================
// Implementation Launcher
// ============================================================================
template<typename T>
void launch_self_attention(std::byte *out, const std::byte *q, const std::byte *k, const std::byte *v,
                           size_t seq_len, size_t n_head, size_t head_dim,
                           size_t total_len, size_t n_kv_head, size_t v_dim,
                           float scale) {
    
    const T *q_ptr = reinterpret_cast<const T *>(q);
    const T *k_ptr = reinterpret_cast<const T *>(k);
    const T *v_ptr = reinterpret_cast<const T *>(v);
    T *o_ptr = reinterpret_cast<T *>(out);

    // Strides for contiguous tensors [seq, head, dim]
    size_t q_s0 = n_head * head_dim, q_s1 = head_dim, q_s2 = 1;
    size_t k_s0 = n_kv_head * head_dim, k_s1 = head_dim, k_s2 = 1;
    size_t v_s0 = n_kv_head * v_dim,   v_s1 = v_dim,   v_s2 = 1;
    size_t o_s0 = n_head * v_dim,      o_s1 = v_dim,   o_s2 = 1;

    size_t group_size = n_head / n_kv_head;
    cublasHandle_t handle = get_cublas_handle();

    // Calculate buffer sizes
    size_t q_need = n_head * seq_len * head_dim * sizeof(float);
    size_t k_need = n_head * total_len * head_dim * sizeof(float);
    size_t v_need = n_head * total_len * v_dim * sizeof(float);
    size_t s_need = n_head * seq_len * total_len * sizeof(float);
    size_t o_need = n_head * seq_len * v_dim * sizeof(float);

    ensure_buf(s_q_all, s_q_all_sz, q_need);
    ensure_buf(s_k_all, s_k_all_sz, k_need);
    ensure_buf(s_v_all, s_v_all_sz, v_need);
    ensure_buf(s_scores, s_scores_sz, s_need);
    ensure_buf(s_o_all, s_o_all_sz, o_need);

    if (!s_q_all || !s_k_all || !s_v_all || !s_scores || !s_o_all) {
        fprintf(stderr, "SelfAttention: Failed to allocate temporary buffers.\n");
        return;
    }

    int thr = 256;

    // 1. Gather Q
    {
        size_t n = n_head * seq_len * head_dim;
        int blk = (int)((n + thr - 1) / thr);
        gather_all_q_kernel<T><<<blk, thr>>>(s_q_all, q_ptr, seq_len, n_head, head_dim, q_s0, q_s1, q_s2);
    }

    // 2. Gather & Expand K
    {
        size_t n = n_head * total_len * head_dim;
        int blk = (int)((n + thr - 1) / thr);
        gather_expand_kv_kernel<T><<<blk, thr>>>(s_k_all, k_ptr, total_len, n_head, n_kv_head, head_dim, group_size, k_s0, k_s1, k_s2);
    }
    // 3. Gather & Expand V
    {
        size_t n = n_head * total_len * v_dim;
        int blk = (int)((n + thr - 1) / thr);
        gather_expand_kv_kernel<T><<<blk, thr>>>(s_v_all, v_ptr, total_len, n_head, n_kv_head, v_dim, group_size, v_s0, v_s1, v_s2);
    }

    // 4. Batched GEMM: Scores = K * Q^T
    {
        float alpha = 1.0f, beta = 0.0f;
        long long strideA = (long long)(total_len * head_dim);
        long long strideB = (long long)(seq_len * head_dim);
        long long strideC = (long long)(seq_len * total_len);
        
        cublasStatus_t status = cublasSgemmStridedBatched(handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            (int)total_len, (int)seq_len, (int)head_dim,
            &alpha,
            s_k_all, (int)head_dim, strideA,
            s_q_all, (int)head_dim, strideB,
            &beta,
            s_scores, (int)total_len, strideC,
            (int)n_head);
        
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "cuBLAS GEMM (QK^T) failed: %d\n", (int)status);
        }
    }

    // 5. Scale + Mask
    {
        size_t ns = n_head * seq_len * total_len;
        int blk = (int)((ns + thr - 1) / thr);
        scale_kernel<<<blk, thr>>>(s_scores, scale, ns);
        causal_mask_batched_kernel<<<blk, thr>>>(s_scores, n_head, seq_len, total_len);
    }

    // 6. Softmax
    {
        size_t num_rows = n_head * seq_len;
        int sthr = 1;
        while (sthr < (int)total_len && sthr < 1024) sthr <<= 1;
        size_t smem = sthr * sizeof(float);
        softmax_row_kernel<<<(int)num_rows, sthr, smem>>>(s_scores, num_rows, total_len);
    }

    // 7. Batched GEMM: Out = Scores * V
    {
        float alpha = 1.0f, beta = 0.0f;
        long long strideA = (long long)(total_len * v_dim);
        long long strideB = (long long)(seq_len * total_len);
        long long strideC = (long long)(seq_len * v_dim);

        cublasStatus_t status = cublasSgemmStridedBatched(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            (int)v_dim, (int)seq_len, (int)total_len,
            &alpha,
            s_v_all, (int)v_dim, strideA,
            s_scores, (int)total_len, strideB,
            &beta,
            s_o_all, (int)v_dim, strideC,
            (int)n_head);

        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "cuBLAS GEMM (SV) failed: %d\n", (int)status);
        }
    }

    // 8. Scatter Output
    {
        size_t n = n_head * seq_len * v_dim;
        int blk = (int)((n + thr - 1) / thr);
        scatter_all_heads_kernel<T><<<blk, thr>>>(o_ptr, s_o_all, seq_len, n_head, v_dim, o_s0, o_s1, o_s2);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA SelfAttention Kernel failed: %s\n", cudaGetErrorString(err));
    }
}

// ============================================================================
// Public API
// ============================================================================
void self_attention(std::byte *out, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t dtype,
                    size_t seq_len, size_t n_head, size_t head_dim,
                    size_t total_len, size_t n_kv_head, size_t v_dim,
                    float scale) {
    
    if (n_head % n_kv_head != 0) {
        fprintf(stderr, "SelfAttention: n_head must be divisible by n_kv_head.\n");
        return;
    }

    switch (dtype) {
        case LLAISYS_DTYPE_F32:
            launch_self_attention<float>(out, q, k, v, seq_len, n_head, head_dim, total_len, n_kv_head, v_dim, scale);
            break;
        case LLAISYS_DTYPE_F16:
            launch_self_attention<half>(out, q, k, v, seq_len, n_head, head_dim, total_len, n_kv_head, v_dim, scale);
            break;
        case LLAISYS_DTYPE_BF16:
            launch_self_attention<nv_bfloat16>(out, q, k, v, seq_len, n_head, head_dim, total_len, n_kv_head, v_dim, scale);
            break;
        default:
            fprintf(stderr, "Unsupported dtype for NVIDIA SelfAttention: %d\n", dtype);
            break;
    }
}

} // namespace llaisys::ops::nvidia