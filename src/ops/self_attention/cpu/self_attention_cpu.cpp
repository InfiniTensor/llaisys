#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace {
// CPU 上的自注意力实现（支持多头、分组查询注意力 GQA）
// 假设输入布局为 [seq_len, num_heads, head_dim]
template <typename T>
void self_attn_impl(std::byte *output,
                    const std::byte *query,
                    const std::byte *key,
                    const std::byte *value,
                    size_t query_len,
                    size_t kv_len,
                    size_t num_q_heads,
                    size_t num_kv_heads,
                    size_t qk_head_dim,
                    size_t v_head_dim,
                    float softmax_scale) {
    const T *q_data = reinterpret_cast<const T *>(query);
    const T *k_data = reinterpret_cast<const T *>(key);
    const T *v_data = reinterpret_cast<const T *>(value);
    T *out_data = reinterpret_cast<T *>(output);

    // Stride 计算：每个序列位置跨越所有头
    const size_t q_seq_stride = num_q_heads * qk_head_dim;
    const size_t k_seq_stride = num_kv_heads * qk_head_dim;
    const size_t v_seq_stride = num_kv_heads * v_head_dim;
    const size_t out_seq_stride = num_q_heads * v_head_dim;

    // 每个头内部的 stride
    const size_t q_head_stride = qk_head_dim;
    const size_t k_head_stride = qk_head_dim;
    const size_t v_head_stride = v_head_dim;
    const size_t out_head_stride = v_head_dim;

    // 分组查询注意力（GQA）因子：每个 KV 头被多少个 Q 头共享
    const int group_size = static_cast<int>(num_q_heads / num_kv_heads);

    // 临时缓冲区用于 logits 和 softmax 概率
    std::vector<float> attention_logits(kv_len);
    std::vector<float> attention_probs(kv_len);

    for (size_t q_pos = 0; q_pos < query_len; ++q_pos) {
        for (size_t q_head = 0; q_head < num_q_heads; ++q_head) {
            const T *q_vec = q_data + q_pos * q_seq_stride + q_head * q_head_stride;

            // 根据 GQA 映射到对应的 KV 头
            int kv_head = static_cast<int>(q_head / group_size);
            const T *k_head_base = k_data + kv_head * k_head_stride;
            const T *v_head_base = v_data + kv_head * v_head_stride;

            // 因果掩码边界：只允许关注到当前 token 及之前（假设 kv_len >= query_len）
            int causal_limit = static_cast<int>(q_pos + kv_len - query_len);

            // Step 1: 计算 QK^T（带因果掩码）
            float max_logit = -std::numeric_limits<float>::infinity();
            for (size_t kv_pos = 0; kv_pos < kv_len; ++kv_pos) {
                float score;
                if (static_cast<int>(kv_pos) > causal_limit) {
                    // 超出因果窗口，置为极小值（近似 -inf）
                    score = -1e20f;
                } else {
                    const T *k_vec = k_head_base + kv_pos * k_seq_stride;
                    float dot_product = 0.0f;
                    for (size_t d = 0; d < qk_head_dim; ++d) {
                        dot_product += llaisys::utils::cast<float>(q_vec[d]) *
                                       llaisys::utils::cast<float>(k_vec[d]);
                    }
                    score = dot_product * softmax_scale;
                }
                attention_logits[kv_pos] = score;
                max_logit = std::max(max_logit, score);
            }

            // Step 2: 稳定化 softmax
            float sum_exp = 0.0f;
            for (size_t kv_pos = 0; kv_pos < kv_len; ++kv_pos) {
                float exp_val = std::exp(attention_logits[kv_pos] - max_logit);
                attention_probs[kv_pos] = exp_val;
                sum_exp += exp_val;
            }
            float inv_sum = 1.0f / sum_exp;

            // Step 3: 加权聚合 V
            T *out_vec = out_data + q_pos * out_seq_stride + q_head * out_head_stride;
            for (size_t d = 0; d < v_head_dim; ++d) {
                float weighted_sum = 0.0f;
                for (size_t kv_pos = 0; kv_pos < kv_len; ++kv_pos) {
                    const T *v_vec = v_head_base + kv_pos * v_seq_stride;
                    weighted_sum += (attention_probs[kv_pos] * inv_sum) *
                                    llaisys::utils::cast<float>(v_vec[d]);
                }
                out_vec[d] = llaisys::utils::cast<T>(weighted_sum);
            }
        }
    }
}
}

namespace llaisys::ops::cpu {

// 调度自注意力操作到 CPU 后端
void self_attention(std::byte *output,
                    const std::byte *query,
                    const std::byte *key,
                    const std::byte *value,
                    llaisysDataType_t dtype,
                    size_t qlen,
                    size_t kvlen,
                    size_t nhead,
                    size_t nkvh,
                    size_t dim,
                    size_t dv,
                    float scale) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return self_attn_impl<float>(
            output, query, key, value, qlen, kvlen, nhead, nkvh, dim, dv, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attn_impl<llaisys::bf16_t>(
            output, query, key, value, qlen, kvlen, nhead, nkvh, dim, dv, scale);
    case LLAISYS_DTYPE_F16:
        return self_attn_impl<llaisys::fp16_t>(
            output, query, key, value, qlen, kvlen, nhead, nkvh, dim, dv, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu