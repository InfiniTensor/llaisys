#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

// Self-Attention: Y = causal_softmax(Q * K^T * scale) * V
template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v,
                     size_t seq_len, size_t total_len, size_t n_heads,
                     size_t n_kv_heads, size_t head_dim, float scale) {
    // 计算每个 KV head 对应多少个 Q head (Grouped Query Attention)
    size_t heads_per_kv = n_heads / n_kv_heads;

    // 为注意力分数分配临时缓冲区
    std::vector<float> attn_scores(seq_len * total_len);

    // 对每个 head 处理
    for (size_t h = 0; h < n_heads; h++) {
        size_t kv_head = h / heads_per_kv; // 对应的 KV head

        // 计算注意力分数: A = Q * K^T * scale
        for (size_t i = 0; i < seq_len; i++) {
            for (size_t j = 0; j < total_len; j++) {
                float score = 0.0f;

                // 点积: Q[i] · K[j]
                for (size_t d = 0; d < head_dim; d++) {
                    size_t q_idx = i * n_heads * head_dim + h * head_dim + d;
                    size_t k_idx = j * n_kv_heads * head_dim + kv_head * head_dim + d;

                    float q_val, k_val;
                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        q_val = llaisys::utils::cast<float>(q[q_idx]);
                        k_val = llaisys::utils::cast<float>(k[k_idx]);
                    } else {
                        q_val = static_cast<float>(q[q_idx]);
                        k_val = static_cast<float>(k[k_idx]);
                    }

                    score += q_val * k_val;
                }

                score *= scale;

                // 应用因果掩码 (causal mask): 只能看到当前及之前的位置
                // 当前查询位置在序列中的绝对位置
                size_t query_pos = total_len - seq_len + i;
                if (j > query_pos) {
                    score = -std::numeric_limits<float>::infinity();
                }

                attn_scores[i * total_len + j] = score;
            }
        }

        // 对每一行应用 softmax
        for (size_t i = 0; i < seq_len; i++) {
            float *row = &attn_scores[i * total_len];

            // 找到最大值（用于数值稳定性）
            float max_score = -std::numeric_limits<float>::infinity();
            for (size_t j = 0; j < total_len; j++) {
                max_score = std::max(max_score, row[j]);
            }

            // 计算 exp 和 sum
            float sum_exp = 0.0f;
            for (size_t j = 0; j < total_len; j++) {
                if (std::isinf(row[j]) && row[j] < 0) {
                    row[j] = 0.0f;
                } else {
                    row[j] = std::exp(row[j] - max_score);
                    sum_exp += row[j];
                }
            }

            // 归一化
            for (size_t j = 0; j < total_len; j++) {
                row[j] /= sum_exp;
            }
        }

        // 计算输出: Y = attention_weights * V
        for (size_t i = 0; i < seq_len; i++) {
            for (size_t d = 0; d < head_dim; d++) {
                float output = 0.0f;

                for (size_t j = 0; j < total_len; j++) {
                    float attn_weight = attn_scores[i * total_len + j];
                    size_t v_idx = j * n_kv_heads * head_dim + kv_head * head_dim + d;

                    float v_val;
                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        v_val = llaisys::utils::cast<float>(v[v_idx]);
                    } else {
                        v_val = static_cast<float>(v[v_idx]);
                    }

                    output += attn_weight * v_val;
                }

                size_t out_idx = i * n_heads * head_dim + h * head_dim + d;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    attn_val[out_idx] = llaisys::utils::cast<T>(output);
                } else {
                    attn_val[out_idx] = static_cast<T>(output);
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t type, size_t seq_len, size_t total_len,
                    size_t n_heads, size_t n_kv_heads, size_t head_dim, float scale) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val),
                               reinterpret_cast<const float *>(q),
                               reinterpret_cast<const float *>(k),
                               reinterpret_cast<const float *>(v),
                               seq_len, total_len, n_heads, n_kv_heads, head_dim, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val),
                               reinterpret_cast<const llaisys::bf16_t *>(q),
                               reinterpret_cast<const llaisys::bf16_t *>(k),
                               reinterpret_cast<const llaisys::bf16_t *>(v),
                               seq_len, total_len, n_heads, n_kv_heads, head_dim, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val),
                               reinterpret_cast<const llaisys::fp16_t *>(q),
                               reinterpret_cast<const llaisys::fp16_t *>(k),
                               reinterpret_cast<const llaisys::fp16_t *>(v),
                               seq_len, total_len, n_heads, n_kv_heads, head_dim, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
