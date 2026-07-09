#include "self_attention_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <type_traits>

namespace llaisys::ops::cpu {

// --- 类型转换辅助 ---
template <typename T>
inline float val_to_float(T v) {
    if constexpr (std::is_same_v<T, float>) {
        return v;
    } else if constexpr (std::is_same_v<T, llaisys::fp16_t>) {
        return llaisys::utils::_f16_to_f32(v);
    } else if constexpr (std::is_same_v<T, llaisys::bf16_t>) {
        return llaisys::utils::_bf16_to_f32(v);
    } else {
        return (float)v;
    }
}

template <typename T>
inline T float_to_val(float v) {
    if constexpr (std::is_same_v<T, float>) {
        return v;
    } else if constexpr (std::is_same_v<T, llaisys::fp16_t>) {
        return llaisys::utils::_f32_to_f16(v);
    } else if constexpr (std::is_same_v<T, llaisys::bf16_t>) {
        return llaisys::utils::_f32_to_bf16(v);
    } else {
        return (T)v;
    }
}

// --- Self Attention 核心计算模板 ---
template<typename T>
void self_attention_kernel(T *attn_val, const T *q, const T *k, const T *v,
                           size_t seqlen, size_t total_len, 
                           size_t nhead, size_t nkvhead, 
                           size_t d, size_t dv,
                           float scale) {
    size_t group_size = nhead / nkvhead;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t task = 0; task < static_cast<ptrdiff_t>(seqlen * nhead); ++task) {
        size_t t = static_cast<size_t>(task) / nhead;
        size_t h = static_cast<size_t>(task) % nhead;
        size_t current_global_pos = total_len - seqlen + t;
        size_t kv_h = h / group_size;

        std::vector<double> scores(total_len, 0.0);
        std::vector<double> acc(dv, 0.0);
        double max_score = -std::numeric_limits<double>::infinity();

        for (size_t pos = 0; pos < total_len; ++pos) {
            if (pos > current_global_pos) {
                scores[pos] = -std::numeric_limits<float>::infinity();
                continue;
            }

            double dot = 0.0;
            size_t q_offset = t * nhead * d + h * d;
            size_t k_offset = pos * nkvhead * d + kv_h * d;
            for (size_t i = 0; i < d; ++i) {
                dot += static_cast<double>(val_to_float(q[q_offset + i])) *
                       static_cast<double>(val_to_float(k[k_offset + i]));
            }

            dot *= static_cast<double>(scale);
            scores[pos] = dot;
            if (dot > max_score) {
                max_score = dot;
            }
        }

        double sum_exp = 0.0;
        for (size_t pos = 0; pos < total_len; ++pos) {
            if (scores[pos] == -std::numeric_limits<double>::infinity()) {
                scores[pos] = 0.0;
                continue;
            }
            scores[pos] = std::exp(scores[pos] - max_score);
            sum_exp += scores[pos];
        }

        double inv_sum = 1.0 / sum_exp;
        for (size_t pos = 0; pos < total_len; ++pos) {
            scores[pos] *= inv_sum;
        }

        for (size_t pos = 0; pos < total_len; ++pos) {
            double weight = scores[pos];
            if (weight == 0.0) {
                continue;
            }

            size_t v_offset = pos * nkvhead * dv + kv_h * dv;
            for (size_t i = 0; i < dv; ++i) {
                acc[i] += weight * static_cast<double>(val_to_float(v[v_offset + i]));
            }
        }

        size_t out_offset = t * nhead * dv + h * dv;
        for (size_t i = 0; i < dv; ++i) {
            attn_val[out_offset + i] = float_to_val<T>(static_cast<float>(acc[i]));
        }
    }
}

// --- 入口分发 ---
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    size_t seqlen, size_t total_len, 
                    size_t nhead, size_t nkvhead, 
                    size_t d, size_t dv,
                    float scale,
                    llaisysDataType_t dtype) {
    
    if (dtype == LLAISYS_DTYPE_F32) {
        self_attention_kernel<float>(
            (float*)attn_val, (const float*)q, (const float*)k, (const float*)v,
            seqlen, total_len, nhead, nkvhead, d, dv, scale
        );
    } else if (dtype == LLAISYS_DTYPE_F16) {
        self_attention_kernel<llaisys::fp16_t>(
            (llaisys::fp16_t*)attn_val, (const llaisys::fp16_t*)q, (const llaisys::fp16_t*)k, (const llaisys::fp16_t*)v,
            seqlen, total_len, nhead, nkvhead, d, dv, scale
        );
    } else if (dtype == LLAISYS_DTYPE_BF16) {
        self_attention_kernel<llaisys::bf16_t>(
            (llaisys::bf16_t*)attn_val, (const llaisys::bf16_t*)q, (const llaisys::bf16_t*)k, (const llaisys::bf16_t*)v,
            seqlen, total_len, nhead, nkvhead, d, dv, scale
        );
    }
}

} // namespace llaisys::ops::cpu
