#include "op.hpp"
#include "../../utils.hpp"
#include <cmath>
#include <vector>
#include <limits>
#include <algorithm>

namespace llaisys::ops {

template <typename T>
void self_attention_cpu(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    size_t seqlen = q->shape()[0];
    size_t nhead = q->shape()[1];
    size_t head_dim = q->shape()[2];
    
    size_t total_len = k->shape()[0];
    size_t nkvhead = k->shape()[1];
    size_t v_dim = v->shape()[2];
    
    size_t group_size = nhead / nkvhead;

    // 计算当前 Q 在 K 中的起始偏移量 (假设 K 包含了 Q，且 Q 位于 K 的末尾)
    // 这是 standard KV Cache 的布局
    size_t q_start_index = total_len - seqlen;

    T* out_ptr = reinterpret_cast<T*>(attn_val->data());
    const T* q_ptr = reinterpret_cast<const T*>(q->data());
    const T* k_ptr = reinterpret_cast<const T*>(k->data());
    const T* v_ptr = reinterpret_cast<const T*>(v->data());

    // 预分配 scores 内存
    std::vector<float> scores(total_len);

    for (size_t s = 0; s < seqlen; ++s) {
        for (size_t h = 0; h < nhead; ++h) {
            size_t kv_h = h / group_size;
            
            const T* q_vec = q_ptr + (s * nhead + h) * head_dim;

            float max_score = -std::numeric_limits<float>::infinity();

            // --- 1. 计算 Attention Scores ---
            for (size_t t = 0; t < total_len; ++t) {
                const T* k_vec = k_ptr + (t * nkvhead + kv_h) * head_dim;

                float dot = 0.0f;
                for (size_t i = 0; i < head_dim; ++i) {
                    dot += utils::cast<float>(q_vec[i]) * utils::cast<float>(k_vec[i]);
                }
                float score = dot * scale;

                // --- 修正后的 Causal Masking ---
                // s 是当前处理的 token 在 Q 中的下标 (0, 1, 2...)
                // absolute_pos 是它在整个 K (历史+当前) 中的绝对下标
                size_t absolute_pos = q_start_index + s;

                // 我们只能看 absolute_pos 及其之前的内容
                if (t > absolute_pos) {
                    score = -std::numeric_limits<float>::infinity();
                }

                scores[t] = score;
                if (score > max_score) max_score = score;
            }

            // --- 2. Softmax ---
            float sum_exp = 0.0f;
            for (size_t t = 0; t < total_len; ++t) {
                if (scores[t] == -std::numeric_limits<float>::infinity()) {
                    scores[t] = 0.0f; 
                } else {
                    float exp_val = std::exp(scores[t] - max_score);
                    scores[t] = exp_val;
                    sum_exp += exp_val;
                }
            }
            
            float inv_sum = 1.0f / (sum_exp + 1e-6f);
            for (size_t t = 0; t < total_len; ++t) {
                scores[t] *= inv_sum;
            }

            // --- 3. Weighted Sum ---
            T* out_vec = out_ptr + (s * nhead + h) * v_dim;
            
            std::vector<float> acc(v_dim, 0.0f);

            for (size_t t = 0; t < total_len; ++t) {
                float prob = scores[t];
                if (prob < 1e-9f) continue;

                const T* v_vec = v_ptr + (t * nkvhead + kv_h) * v_dim;
                for (size_t i = 0; i < v_dim; ++i) {
                    acc[i] += prob * utils::cast<float>(v_vec[i]);
                }
            }

            for (size_t i = 0; i < v_dim; ++i) {
                out_vec[i] = utils::cast<T>(acc[i]);
            }
        }
    }
}

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    switch (q->dtype()) {
        case LLAISYS_DTYPE_F32:
            self_attention_cpu<float>(attn_val, q, k, v, scale);
            break;
        case LLAISYS_DTYPE_F16:
            self_attention_cpu<fp16_t>(attn_val, q, k, v, scale);
            break;
        case LLAISYS_DTYPE_BF16:
            self_attention_cpu<bf16_t>(attn_val, q, k, v, scale);
            break;
        default:
            break;
    }
}

} // namespace llaisys::ops