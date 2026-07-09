#include "self_attention_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <vector>
#include <algorithm>

template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v,
                     size_t seq_len, size_t total_len,
                     size_t n_heads, size_t n_kv_heads, size_t head_dim, size_t v_dim, float scale) {
    // q: (seq_len, n_heads, head_dim)
    // k: (total_len, n_kv_heads, head_dim)
    // v: (total_len, n_kv_heads, v_dim)
    // attn_val: (seq_len, n_heads, v_dim)

    // Group query attention: each kv head serves multiple q heads
    size_t heads_per_kv = n_heads / n_kv_heads;

    for (size_t s = 0; s < seq_len; s++) {
        for (size_t h = 0; h < n_heads; h++) {
            // Which KV head to use
            size_t kv_head = h / heads_per_kv;

            // Calculate attention scores: Q @ K^T * scale
            std::vector<float> scores(total_len);
            for (size_t t = 0; t < total_len; t++) {
                float score = 0.0f;
                for (size_t d = 0; d < head_dim; d++) {
                    float q_val, k_val;
                    size_t q_idx = s * n_heads * head_dim + h * head_dim + d;
                    size_t k_idx = t * n_kv_heads * head_dim + kv_head * head_dim + d;

                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        q_val = llaisys::utils::cast<float>(q[q_idx]);
                        k_val = llaisys::utils::cast<float>(k[k_idx]);
                    } else {
                        q_val = static_cast<float>(q[q_idx]);
                        k_val = static_cast<float>(k[k_idx]);
                    }
                    score += q_val * k_val;
                }
                scores[t] = score * scale;
            }

            // Apply causal mask and softmax
            // Causal mask: only attend to positions <= current_position
            // current_position in full context is: total_len - seq_len + s
            size_t current_pos = total_len - seq_len + s;

            // Find max for numerical stability
            float max_score = -INFINITY;
            for (size_t t = 0; t <= current_pos; t++) {
                max_score = std::max(max_score, scores[t]);
            }

            // Compute exp and sum
            float exp_sum = 0.0f;
            for (size_t t = 0; t <= current_pos; t++) {
                scores[t] = std::exp(scores[t] - max_score);
                exp_sum += scores[t];
            }

            // Normalize
            for (size_t t = 0; t <= current_pos; t++) {
                scores[t] /= exp_sum;
            }

            // Set masked positions to 0
            for (size_t t = current_pos + 1; t < total_len; t++) {
                scores[t] = 0.0f;
            }

            // Multiply with V: scores @ V
            for (size_t d = 0; d < v_dim; d++) {
                float sum = 0.0f;
                for (size_t t = 0; t < total_len; t++) {
                    float v_val;
                    size_t v_idx = t * n_kv_heads * v_dim + kv_head * v_dim + d;

                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        v_val = llaisys::utils::cast<float>(v[v_idx]);
                    } else {
                        v_val = static_cast<float>(v[v_idx]);
                    }
                    sum += scores[t] * v_val;
                }

                size_t out_idx = s * n_heads * v_dim + h * v_dim + d;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    attn_val[out_idx] = llaisys::utils::cast<T>(sum);
                } else {
                    attn_val[out_idx] = static_cast<T>(sum);
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t dtype, size_t seq_len, size_t total_len,
                    size_t n_heads, size_t n_kv_heads, size_t head_dim, size_t v_dim, float scale) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return self_attention_<float>(
            reinterpret_cast<float *>(attn_val),
            reinterpret_cast<const float *>(q),
            reinterpret_cast<const float *>(k),
            reinterpret_cast<const float *>(v),
            seq_len, total_len, n_heads, n_kv_heads, head_dim, v_dim, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_<llaisys::bf16_t>(
            reinterpret_cast<llaisys::bf16_t *>(attn_val),
            reinterpret_cast<const llaisys::bf16_t *>(q),
            reinterpret_cast<const llaisys::bf16_t *>(k),
            reinterpret_cast<const llaisys::bf16_t *>(v),
            seq_len, total_len, n_heads, n_kv_heads, head_dim, v_dim, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_<llaisys::fp16_t>(
            reinterpret_cast<llaisys::fp16_t *>(attn_val),
            reinterpret_cast<const llaisys::fp16_t *>(q),
            reinterpret_cast<const llaisys::fp16_t *>(k),
            reinterpret_cast<const llaisys::fp16_t *>(v),
            seq_len, total_len, n_heads, n_kv_heads, head_dim, v_dim, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
