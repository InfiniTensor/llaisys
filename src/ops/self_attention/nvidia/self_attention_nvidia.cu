#include "self_attention_nvidia.cuh"

#include "../../../utils.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cfloat>
#include <cstdio>
#include <stdexcept>
#include <type_traits>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void self_attention_kernel_impl(T *attn_val, const T *q, const T *k, const T *v,
                                           float scale, size_t seq_len, size_t total_len,
                                           size_t nhead, size_t nkvhead, size_t d, size_t dv) {
    if (threadIdx.x != 0) {return;}

    extern __shared__ double scores[];

    size_t query_idx = blockIdx.x;
    size_t head_idx = blockIdx.y;
    size_t n_rep = nhead / nkvhead;
    size_t kv_head_idx = head_idx / n_rep;

    size_t global_pos = total_len - seq_len + query_idx;
    size_t valid_len = global_pos + 1;
    if (valid_len > total_len) {
        valid_len = total_len;
    }

    const T *q_ptr = q + (query_idx * nhead * d) + (head_idx * d);
    T *out_ptr = attn_val + (query_idx * nhead * dv) + (head_idx * dv);

    double max_val = -DBL_MAX;
    for (size_t t = 0; t < valid_len; ++t) {
        const T *k_ptr = k + (t * nkvhead * d) + (kv_head_idx * d);
        double dot = 0.0;
        for (size_t j = 0; j < d; ++j) {
            if constexpr (std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t>) {
                dot += static_cast<double>(utils::cast_device<float>(q_ptr[j])) *
                       static_cast<double>(utils::cast_device<float>(k_ptr[j]));
            } else {
                dot += static_cast<double>(q_ptr[j]) * static_cast<double>(k_ptr[j]);
            }
        }
        scores[t] = dot * static_cast<double>(scale);
        if (scores[t] > max_val) {
            max_val = scores[t];
        }
    }

    double sum_exp = 0.0;
    for (size_t t = 0; t < valid_len; ++t) {
        scores[t] = exp(scores[t] - max_val);
        sum_exp += scores[t];
    }

    for (size_t j = 0; j < dv; ++j) {
        double value = 0.0;
        for (size_t t = 0; t < valid_len; ++t) {
            const T *v_ptr = v + (t * nkvhead * dv) + (kv_head_idx * dv);
            if constexpr (std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t>) {
                value += scores[t] * static_cast<double>(utils::cast_device<float>(v_ptr[j]));
            } else {
                value += scores[t] * static_cast<double>(v_ptr[j]);
            }
        }
        if constexpr (std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t>) {
            out_ptr[j] = utils::cast_device<T>(static_cast<float>(value / sum_exp));
        } else {
            out_ptr[j] = static_cast<T>(value / sum_exp);
        }
    }
}

void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    float scale, llaisysDataType_t dtype,
                    size_t seq_len, size_t total_len, size_t nhead, size_t nkvhead, size_t d, size_t dv) {
    const dim3 num_blocks(static_cast<unsigned int>(seq_len), static_cast<unsigned int>(nhead), 1);
    const int block_size = 1;
    const size_t shared_mem = total_len * sizeof(double);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        self_attention_kernel_impl<<<num_blocks, block_size, shared_mem>>>(
            reinterpret_cast<float *>(attn_val),
            reinterpret_cast<const float *>(q),
            reinterpret_cast<const float *>(k),
            reinterpret_cast<const float *>(v),
            scale, seq_len, total_len, nhead, nkvhead, d, dv);
        break;
    case LLAISYS_DTYPE_F16:
        self_attention_kernel_impl<<<num_blocks, block_size, shared_mem>>>(
            reinterpret_cast<fp16_t *>(attn_val),
            reinterpret_cast<const fp16_t *>(q),
            reinterpret_cast<const fp16_t *>(k),
            reinterpret_cast<const fp16_t *>(v),
            scale, seq_len, total_len, nhead, nkvhead, d, dv);
        break;
    case LLAISYS_DTYPE_BF16:
        self_attention_kernel_impl<<<num_blocks, block_size, shared_mem>>>(
            reinterpret_cast<bf16_t *>(attn_val),
            reinterpret_cast<const bf16_t *>(q),
            reinterpret_cast<const bf16_t *>(k),
            reinterpret_cast<const bf16_t *>(v),
            scale, seq_len, total_len, nhead, nkvhead, d, dv);
        break;
    default:
        std::fprintf(stderr, "[ERROR] Unsupported data type for CUDA self_attention: %d\n", dtype);
        throw std::runtime_error("Unsupported data type");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[CUDA ERROR] self_attention kernel launch failed: %s\n", cudaGetErrorString(err));
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

} // namespace llaisys::ops::nvidia
