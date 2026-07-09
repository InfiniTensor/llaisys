#include "self_attention_nvidia.hpp"

#include "../../nvidia_cuda.cuh"

#include <cmath>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void self_attention_kernel(
    T *attn_val,
    const T *q,
    const T *k,
    const T *v,
    size_t seqlen,
    size_t total_len,
    size_t nhead,
    size_t nkvhead,
    size_t d,
    size_t dv,
    float scale) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = seqlen * nhead * dv;
    if (idx >= total) {
        return;
    }

    size_t cdv = idx % dv;
    size_t t = idx / dv;
    size_t h = t % nhead;
    size_t i = t / nhead;

    size_t n_groups = nhead / nkvhead;
    size_t h_kv = h / n_groups;

    ptrdiff_t cutoff = static_cast<ptrdiff_t>(total_len) - static_cast<ptrdiff_t>(seqlen) + static_cast<ptrdiff_t>(i);

    float max_score = -INFINITY;
    for (size_t j = 0; j < total_len; ++j) {
        if (static_cast<ptrdiff_t>(j) > cutoff) {
            continue;
        }

        float dot = 0.0f;
        size_t q_base = (i * nhead + h) * d;
        size_t k_base = (j * nkvhead + h_kv) * d;
        for (size_t c = 0; c < d; ++c) {
            dot += to_float(q[q_base + c]) * to_float(k[k_base + c]);
        }
        float score = dot * scale;
        max_score = fmaxf(max_score, score);
    }

    float sum_exp = 0.0f;
    for (size_t j = 0; j < total_len; ++j) {
        if (static_cast<ptrdiff_t>(j) > cutoff) {
            continue;
        }

        float dot = 0.0f;
        size_t q_base = (i * nhead + h) * d;
        size_t k_base = (j * nkvhead + h_kv) * d;
        for (size_t c = 0; c < d; ++c) {
            dot += to_float(q[q_base + c]) * to_float(k[k_base + c]);
        }
        float score = dot * scale;
        sum_exp += expf(score - max_score);
    }

    float out_val = 0.0f;
    for (size_t j = 0; j < total_len; ++j) {
        if (static_cast<ptrdiff_t>(j) > cutoff) {
            continue;
        }

        float dot = 0.0f;
        size_t q_base = (i * nhead + h) * d;
        size_t k_base = (j * nkvhead + h_kv) * d;
        for (size_t c = 0; c < d; ++c) {
            dot += to_float(q[q_base + c]) * to_float(k[k_base + c]);
        }
        float score = dot * scale;
        float prob = expf(score - max_score) / sum_exp;

        size_t v_idx = (j * nkvhead + h_kv) * dv + cdv;
        out_val += prob * to_float(v[v_idx]);
    }

    attn_val[idx] = from_float<T>(out_val);
}

template <typename T>
void launch_self_attention(
    std::byte *attn_val,
    const std::byte *q,
    const std::byte *k,
    const std::byte *v,
    size_t seqlen,
    size_t total_len,
    size_t nhead,
    size_t nkvhead,
    size_t d,
    size_t dv,
    float scale) {
    size_t total = seqlen * nhead * dv;
    const int threads = num_threads_1d();
    const int blocks = num_blocks_1d(total, threads);
    auto stream = current_stream();
    self_attention_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<T *>(attn_val),
        reinterpret_cast<const T *>(q),
        reinterpret_cast<const T *>(k),
        reinterpret_cast<const T *>(v),
        seqlen,
        total_len,
        nhead,
        nkvhead,
        d,
        dv,
        scale);
    check_cuda(cudaGetLastError(), "self_attention_kernel launch");
}

void self_attention(
    std::byte *attn_val,
    const std::byte *q,
    const std::byte *k,
    const std::byte *v,
    llaisysDataType_t type,
    size_t seqlen,
    size_t total_len,
    size_t nhead,
    size_t nkvhead,
    size_t d,
    size_t dv,
    float scale) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launch_self_attention<float>(attn_val, q, k, v, seqlen, total_len, nhead, nkvhead, d, dv, scale);
    case LLAISYS_DTYPE_F16:
        return launch_self_attention<llaisys::fp16_t>(attn_val, q, k, v, seqlen, total_len, nhead, nkvhead, d, dv, scale);
    case LLAISYS_DTYPE_BF16:
        return launch_self_attention<llaisys::bf16_t>(attn_val, q, k, v, seqlen, total_len, nhead, nkvhead, d, dv, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia
