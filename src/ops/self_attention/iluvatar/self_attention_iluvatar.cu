#include "self_attention_iluvatar.cuh"

#include "../../../device/iluvatar/cuda_utils.cuh"
#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

namespace llaisys::ops::iluvatar {

// Compute Q * K^T for one attention head
template <typename T>
__global__ void qkT_kernel(const T *q, const T *k, float *scores,
                           size_t seq_len, size_t total_len, size_t d,
                           size_t nhead, size_t nkvhead, float scale) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_scores = seq_len * total_len;

    if (idx >= total_scores) return;

    size_t j = idx % total_len;
    size_t i = idx / total_len;

    size_t num_repeats = nhead / nkvhead;

    float dot_product = 0.0f;
    for (size_t k_idx = 0; k_idx < d; ++k_idx) {
        size_t q_pos = i * nhead * d + blockIdx.y * d + k_idx;
        float q_val = to_float_cuda(q[q_pos]);

        size_t kvh = blockIdx.y / num_repeats;
        size_t k_pos = j * nkvhead * d + kvh * d + k_idx;
        float k_val = to_float_cuda(k[k_pos]);

        dot_product += q_val * k_val;
    }

    scores[idx * nhead + blockIdx.y] = dot_product * scale;
}

// Apply causal mask and softmax
template <typename T>
__global__ void causal_softmax_kernel(float *scores, T *attn_weights,
                                      size_t seq_len, size_t total_len, size_t nhead) {
    size_t i = blockIdx.x;
    size_t h = blockIdx.y;

    size_t diagonal = total_len - seq_len;

    extern __shared__ float shared_mem[];
    float *sdata = shared_mem;
    float *smax = &shared_mem[blockDim.x];

    float thread_max = -FLT_MAX;
    for (size_t j = threadIdx.x; j < total_len; j += blockDim.x) {
        size_t idx = (i * total_len + j) * nhead + h;
        float val = scores[idx];

        if (j > i + diagonal) {
            val = -FLT_MAX;
        }
        sdata[j] = val;
        if (val > thread_max) thread_max = val;
    }
    __syncthreads();

    smax[threadIdx.x] = thread_max;
    __syncthreads();
    
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (smax[threadIdx.x + s] > smax[threadIdx.x]) {
                smax[threadIdx.x] = smax[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    float max_val = smax[0];

    float thread_sum = 0.0f;
    for (size_t j = threadIdx.x; j < total_len; j += blockDim.x) {
        float exp_val = expf(sdata[j] - max_val);
        sdata[j] = exp_val;
        thread_sum += exp_val;
    }
    __syncthreads();

    __shared__ float ssum[256];
    ssum[threadIdx.x] = thread_sum;
    __syncthreads();
    
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            ssum[threadIdx.x] += ssum[threadIdx.x + s];
        }
        __syncthreads();
    }
    float sum_val = ssum[0];

    for (size_t j = threadIdx.x; j < total_len; j += blockDim.x) {
        size_t idx = (i * total_len + j) * nhead + h;
        scores[idx] = sdata[j] / sum_val;
    }
}

// Compute attention output
template <typename T>
__global__ void attn_v_kernel(const float *attn_weights, const T *v, T *out,
                              size_t seq_len, size_t total_len, size_t nhead, size_t nkvhead, size_t dv) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_outputs = seq_len * nhead * dv;

    if (idx >= total_outputs) return;

    size_t tmp = idx;
    size_t v_idx = tmp % dv;
    tmp /= dv;
    size_t h = tmp % nhead;
    size_t i = tmp / nhead;

    size_t num_repeats = nhead / nkvhead;
    size_t kvh = h / num_repeats;

    float weighted_sum = 0.0f;
    for (size_t j = 0; j < total_len; ++j) {
        size_t attn_idx = (i * total_len + j) * nhead + h;
        float w = attn_weights[attn_idx];

        size_t v_pos = j * nkvhead * dv + kvh * dv + v_idx;
        float v_val = to_float_cuda(v[v_pos]);

        weighted_sum += w * v_val;
    }

    size_t out_pos = i * nhead * dv + h * dv + v_idx;
    out[out_pos] = from_float_cuda<T>(weighted_sum);
}

template <typename T>
void self_attention_(std::byte *attn_val_bytes, const std::byte *q_bytes,
                     const std::byte *k_bytes, const std::byte *v_bytes,
                     float scale, size_t seq_len, size_t total_len,
                     size_t nhead, size_t nkvhead, size_t d, size_t dv) {
    auto attn_val = reinterpret_cast<T *>(attn_val_bytes);
    auto q = reinterpret_cast<const T *>(q_bytes);
    auto k = reinterpret_cast<const T *>(k_bytes);
    auto v = reinterpret_cast<const T *>(v_bytes);

    size_t scores_size = seq_len * total_len * nhead * sizeof(float);
    float *d_scores;
    cudaMalloc(&d_scores, scores_size);

    size_t total_scores = seq_len * total_len;
    const int blockSize = 256;
    const int numBlocks = (total_scores + blockSize - 1) / blockSize;

    dim3 qk_grid(numBlocks, nhead);
    qkT_kernel<T><<<qk_grid, blockSize>>>(q, k, d_scores, seq_len, total_len, d, nhead, nkvhead, scale);

    dim3 softmax_grid(seq_len, nhead);
    size_t shared_mem_size = total_len * sizeof(float) + blockSize * sizeof(float);
    causal_softmax_kernel<T><<<softmax_grid, blockSize, shared_mem_size>>>(
        d_scores, attn_val, seq_len, total_len, nhead);

    size_t total_outputs = seq_len * nhead * dv;
    const int out_numBlocks = (total_outputs + blockSize - 1) / blockSize;
    attn_v_kernel<T><<<out_numBlocks, blockSize>>>(
        d_scores, v, attn_val, seq_len, total_len, nhead, nkvhead, dv);

    cudaFree(d_scores);
}

void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t type, float scale, size_t seq_len, size_t total_len,
                    size_t nhead, size_t nkvhead, size_t d, size_t dv) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_<float>(attn_val, q, k, v, scale, seq_len, total_len, nhead, nkvhead, d, dv);
    case LLAISYS_DTYPE_BF16:
        return self_attention_<bf16_t_cuda>(attn_val, q, k, v, scale, seq_len, total_len, nhead, nkvhead, d, dv);
    case LLAISYS_DTYPE_F16:
        return self_attention_<fp16_t_cuda>(attn_val, q, k, v, scale, seq_len, total_len, nhead, nkvhead, d, dv);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

}
