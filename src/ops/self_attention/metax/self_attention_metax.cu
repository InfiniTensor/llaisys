#include "self_attention_metax.cuh"

#include "../../../device/metax/cuda_utils.cuh"
#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <float.h>

namespace llaisys::ops::metax {

// Kernel to compute Q*K^T and apply causal mask
template <typename T>
__global__ void qkKernel(T *attn_scores, const T *q, const T *k, float scale,
                         size_t seqlen, size_t nhead, size_t nkvhead, size_t d, size_t total_len) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = seqlen * nhead * total_len;
    if (idx >= total) return;

    size_t i = idx / (nhead * total_len);
    size_t h = (idx / total_len) % nhead;
    size_t j = idx % total_len;

    size_t kv_offset = total_len - seqlen;
    if (j > i + kv_offset) {
        attn_scores[idx] = from_float_metax<T>(-FLT_MAX);
        return;
    }

    size_t kv_h = h / (nhead / nkvhead);

    // Blocked accumulation for better precision
    float sum = 0.0f;
    const size_t BLOCK_SIZE = 64;
    for (size_t dim_start = 0; dim_start < d; dim_start += BLOCK_SIZE) {
        float block_sum = 0.0f;
        size_t dim_end = min(dim_start + BLOCK_SIZE, d);
        for (size_t dim = dim_start; dim < dim_end; ++dim) {
            float q_val = to_float_metax(q[i * nhead * d + h * d + dim]);
            float k_val = to_float_metax(k[j * nkvhead * d + kv_h * d + dim]);
            block_sum += q_val * k_val;
        }
        sum += block_sum;
    }

    attn_scores[idx] = from_float_metax<T>(sum * scale);
}

// Kernel for softmax (per row)
template <typename T>
__global__ void softmaxKernel(T *attn_scores, size_t seqlen, size_t nhead, size_t total_len) {
    size_t row = blockIdx.x;
    size_t tid = threadIdx.x;

    if (row >= seqlen * nhead) return;

    T *row_ptr = attn_scores + row * total_len;
    extern __shared__ float shared_mem[];
    float *shared_max = shared_mem;
    float *shared_sum = shared_mem + blockDim.x;

    float local_max = -FLT_MAX;
    for (size_t i = tid; i < total_len; i += blockDim.x) {
        float val = to_float_metax(row_ptr[i]);
        if (val != -FLT_MAX) {
            local_max = fmaxf(local_max, val);
        }
    }
    shared_max[tid] = local_max;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }
    float row_max = shared_max[0];

    float local_sum = 0.0f;
    for (size_t i = tid; i < total_len; i += blockDim.x) {
        float val = to_float_metax(row_ptr[i]);
        if (val != -FLT_MAX) {
            float exp_val = expf(val - row_max);
            local_sum += exp_val;
            row_ptr[i] = from_float_metax<T>(exp_val);
        } else {
            row_ptr[i] = from_float_metax<T>(0.0f);
        }
    }
    shared_sum[tid] = local_sum;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    float row_sum = shared_sum[0];

    for (size_t i = tid; i < total_len; i += blockDim.x) {
        float val = to_float_metax(row_ptr[i]);
        row_ptr[i] = from_float_metax<T>(val / row_sum);
    }
}

// Kernel for attention * V with blocked accumulation
template <typename T>
__global__ void attnVKernel(T *attn_val, const T *attn_scores, const T *v,
                            size_t seqlen, size_t nhead, size_t nkvhead, size_t dv, size_t total_len) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = seqlen * nhead * dv;
    if (idx >= total) return;

    size_t i = idx / (nhead * dv);
    size_t h = (idx / dv) % nhead;
    size_t dim = idx % dv;
    size_t kv_h = h / (nhead / nkvhead);

    // Blocked accumulation for better precision
    float sum = 0.0f;
    const size_t BLOCK_SIZE = 64;
    for (size_t j_start = 0; j_start < total_len; j_start += BLOCK_SIZE) {
        float block_sum = 0.0f;
        size_t j_end = min(j_start + BLOCK_SIZE, total_len);
        for (size_t j = j_start; j < j_end; ++j) {
            float score = to_float_metax(attn_scores[i * nhead * total_len + h * total_len + j]);
            float v_val = to_float_metax(v[j * nkvhead * dv + kv_h * dv + dim]);
            block_sum += score * v_val;
        }
        sum += block_sum;
    }

    attn_val[idx] = from_float_metax<T>(sum);
}

template <typename T>
void self_attention_(std::byte *attn_val_bytes, const std::byte *q_bytes, const std::byte *k_bytes,
                     const std::byte *v_bytes, float scale, size_t seqlen, size_t nhead,
                     size_t nkvhead, size_t d, size_t dv, size_t total_len) {
    auto attn_val = reinterpret_cast<T *>(attn_val_bytes);
    auto q = reinterpret_cast<const T *>(q_bytes);
    auto k = reinterpret_cast<const T *>(k_bytes);
    auto v = reinterpret_cast<const T *>(v_bytes);

    T *attn_scores;
    size_t scores_size = seqlen * nhead * total_len * sizeof(T);
    cudaMalloc(&attn_scores, scores_size);

    const int blockSize = 256;

    size_t qk_total = seqlen * nhead * total_len;
    int qk_blocks = (qk_total + blockSize - 1) / blockSize;
    qkKernel<T><<<qk_blocks, blockSize>>>(attn_scores, q, k, scale, seqlen, nhead, nkvhead, d, total_len);

    size_t softmax_shared_mem = 2 * blockSize * sizeof(float);
    softmaxKernel<T><<<seqlen * nhead, blockSize, softmax_shared_mem>>>(attn_scores, seqlen, nhead, total_len);

    size_t attn_v_total = seqlen * nhead * dv;
    int attn_v_blocks = (attn_v_total + blockSize - 1) / blockSize;
    attnVKernel<T><<<attn_v_blocks, blockSize>>>(attn_val, attn_scores, v, seqlen, nhead, nkvhead, dv, total_len);

    cudaFree(attn_scores);
}

void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t type, float scale, size_t seqlen, size_t nhead, size_t nkvhead,
                    size_t d, size_t dv, size_t total_len) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_<float>(attn_val, q, k, v, scale, seqlen, nhead, nkvhead, d, dv, total_len);
    case LLAISYS_DTYPE_BF16:
        return self_attention_<bf16_t_metax>(attn_val, q, k, v, scale, seqlen, nhead, nkvhead, d, dv, total_len);
    case LLAISYS_DTYPE_F16:
        return self_attention_<fp16_t_metax>(attn_val, q, k, v, scale, seqlen, nhead, nkvhead, d, dv, total_len);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

}
