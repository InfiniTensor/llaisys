#include "self_attension_cpu.hpp"
#include "../../../utils.hpp"
#include <vector>
#include <limits>
#include <cmath>

template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v, size_t seqlen, size_t total_len, size_t nhead, size_t nkv_head, size_t d, size_t dv, float scale) {
    size_t group_size = nhead / nkv_head;
    std::vector<float> scores_buffer(total_len);
    std::fill(attn_val, attn_val + seqlen * nhead * dv, llaisys::utils::cast<T>(0));

    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        for (size_t i = 0; i < seqlen; i++) {
            for (size_t h = 0; h < nhead; h++) {
                size_t group_id = h / group_size;

                float max_score = -std::numeric_limits<float>::infinity();
                const size_t max_t = i + (total_len - seqlen);

                if (max_t < 0) {
                    continue;
                }

                const size_t t_end = std::min<size_t>(max_t, total_len - 1);

                for (size_t t = 0; t <= t_end; t++) {
                    size_t q_base = (i * nhead + h) * d;
                    size_t k_base = (t * nkv_head + group_id) * d;

                    float score = 0.0f;
                    for (size_t j = 0; j < d; j++) {
                        score += llaisys::utils::cast<float>(q[q_base + j]) * llaisys::utils::cast<float>(k[k_base + j]);
                    }
                    score *= scale;
                    scores_buffer[t] = score;

                    if (score > max_score) {
                        max_score = score;
                    }
                }

                // Mask out invalid positions to match causal attention.
                for (size_t t = t_end + 1; t < total_len; t++) {
                    scores_buffer[t] = -std::numeric_limits<float>::infinity();
                }

                // compute softmax
                float sum_exp = 0.0f;
                for (size_t t = 0; t < total_len; t++) {
                    scores_buffer[t] = std::exp(scores_buffer[t] - max_score);
                    sum_exp += scores_buffer[t];
                }
                for (size_t t = 0; t < total_len; t++) {
                    scores_buffer[t] /= sum_exp;
                }

                size_t attn_base = (i * nhead + h) * dv;
                for (size_t j = 0; j < dv; j++) {
                    float out = 0.0f;
                    for (size_t t = 0; t < total_len; t++) {
                        size_t v_base = (t * nkv_head + group_id) * dv;
                        out += scores_buffer[t] * llaisys::utils::cast<float>(v[v_base + j]);
                    }
                    attn_val[attn_base + j] = llaisys::utils::cast<T>(out);
                }
            }
        }
    } else {
        for (size_t i = 0; i < seqlen; i++) {
            for (size_t h = 0; h < nhead; h++) {
                size_t group_id = h / group_size;

                float max_score = -std::numeric_limits<float>::infinity();
                const size_t max_t = i + (total_len - seqlen);

                if (max_t < 0) {
                    continue;
                }

                size_t q_base = (i * nhead + h) * d;
                for (size_t t = 0; t < total_len; t++) {
                    size_t k_base = (t * nkv_head + group_id) * d;

                    float score = 0;

                    for (size_t j = 0; j < d; j++) {
                        score += q[q_base + j] * k[k_base + j];
                    }
                    score *= scale;
                    scores_buffer[t] = score;

                    if (score > max_score) {
                        max_score = score;
                    }
                }

                // Mask out invalid positions to match causal attention.
                for (size_t t = max_t + 1; t < total_len; t++) {
                    scores_buffer[t] = -std::numeric_limits<float>::infinity();
                }

                // compute softmax
                float sum_exp = 0.0f;
                for (size_t t = 0; t < total_len; t++) {
                    scores_buffer[t] = std::exp(scores_buffer[t] - max_score);
                    sum_exp += scores_buffer[t];
                }
                for (size_t t = 0; t < total_len; t++) {
                    scores_buffer[t] /= sum_exp;
                }

                size_t attn_base = (i * nhead + h) * dv;
                for (size_t t = 0; t < total_len; t++) {
                    size_t v_base = (t * nkv_head + group_id) * dv;

                    for (size_t j = 0; j < dv; j++) {
                        attn_val[attn_base + j] += scores_buffer[t] * v[v_base + j];
                    }
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, llaisysDataType_t dtype, size_t seqlen, size_t total_len, size_t nhead, size_t kv_head, size_t d, size_t dv, float scale) {
    switch (dtype) {
    case llaisysDataType_t::LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val), reinterpret_cast<const float *>(q), reinterpret_cast<const float *>(k), reinterpret_cast<const float *>(v), seqlen, total_len, nhead, kv_head, d, dv, scale);
    case llaisysDataType_t::LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val), reinterpret_cast<const llaisys::fp16_t *>(q), reinterpret_cast<const llaisys::fp16_t *>(k), reinterpret_cast<const llaisys::fp16_t *>(v), seqlen, total_len, nhead, kv_head, d, dv, scale);
    case llaisysDataType_t::LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val), reinterpret_cast<const llaisys::bf16_t *>(q), reinterpret_cast<const llaisys::bf16_t *>(k), reinterpret_cast<const llaisys::bf16_t *>(v), seqlen, total_len, nhead, kv_head, d, dv, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
