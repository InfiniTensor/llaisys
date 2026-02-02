#include "self_attention_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <vector>
#include <algorithm>

namespace llaisys::ops::cpu {

template <typename T>
void self_attention_kernel(T *attn_val, const T *q, const T *k, const T *v,
                           size_t seqlen, size_t total_len, 
                           size_t nhead, size_t nkvhead, size_t d, size_t dv, float scale) {
    
    size_t group_size = nhead / nkvhead;
    // past_len 计算的是在当前推理步之前已经存在的 Token 数量。
    size_t past_len = total_len - seqlen;

    // 并行化 Query Token 和 Head 维度
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < seqlen; ++i) {// i-表示第i个token
        for (size_t h = 0; h < nhead; ++h) {// h-表示第h个head
            // 这是 GQA 的核心逻辑。对于第 h 个 Query 头，它需要找到对应的 Key/Value 头索引。
            // 例如，如果 group_size=4，那么 Query 头 0, 1, 2, 3 都对应 KV 头 0。
            size_t kv_h = h / group_size; // GQA 映射
            
            // 定位当前 Q，定位到当前处理的 Query 向量的起始地址
            const T *q_ptr = q + (i * nhead * d) + (h * d);
            
            // 临时存储当前 Query 对所有 Key 的分数
            std::vector<float> scores(total_len);
            float max_score = -INFINITY;

            // 1. 计算点积分数 + Causal Mask
            // 这是实现 Causal Mask (因果掩码) 的关键。对于序列中的第 i 个 Token，
            // 它只能看到它自己 (i) 以及它之前的 Token (past_len + 0 到 i-1)。
            size_t can_see_len = past_len + i + 1;

            for (size_t j = 0; j < total_len; ++j) {
                if (j < can_see_len) {
                    const T *k_ptr = k + (j * nkvhead * d) + (kv_h * d);
                    float sum = 0.0f;
                    for (size_t p = 0; p < d; ++p) { // 内层循环 p 计算 Query 向量和 Key 向量的点积。
                        // 注意这里使用了 kv_h 来定位 Key 向量，体现了 GQA 的共享机制。
                        sum += llaisys::utils::cast<float>(q_ptr[p]) * 
                               llaisys::utils::cast<float>(k_ptr[p]);
                    }
                    scores[j] = sum * scale;
                    if (scores[j] > max_score) max_score = scores[j];
                } else {
                    // 如果 j 超出了可见范围，直接将分数设为负无穷 (-INFINITY)。这样在后续 Softmax 中，它们的概率会变为 0。
                    scores[j] = -INFINITY;  // Mask 掉未来的 Token
                }
            }

            // 2. Softmax-Softmax(Xi)=exp(xi-max_xi) / sum(exp(xi))
            // 减去 max_score 可以防止 std::exp 计算结果溢出 (Overflow)，这是深度学习中的标准操作。
            float exp_sum = 0.0f;
            for (size_t j = 0; j < can_see_len; ++j) {
                scores[j] = std::exp(scores[j] - max_score);
                exp_sum += scores[j];
            }
            for (size_t j = 0; j < can_see_len; ++j) {
                scores[j] /= exp_sum;
            }

            // 3. 加权求和V
            // Output = Σ (Score * V)
            T *out_ptr = attn_val + (i * nhead * dv) + (h * dv);
            for (size_t p = 0; p < dv; ++p) {
                float res = 0.0f;
                for (size_t j = 0; j < can_see_len; ++j) {
                    const T *v_ptr = v + (j * nkvhead * dv) + (kv_h * dv);// 权重V
                    res += scores[j] * llaisys::utils::cast<float>(v_ptr[p]);
                }
                out_ptr[p] = llaisys::utils::cast<T>(res);
            }
        }
    }
}

void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t dtype, size_t seqlen, size_t total_len, 
                    size_t nhead, size_t nkvhead, size_t d, size_t dv, float scale) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        self_attention_kernel<float>((float *)attn_val, (const float *)q, (const float *)k, (const float *)v,
                                     seqlen, total_len, nhead, nkvhead, d, dv, scale);
        break;
    case LLAISYS_DTYPE_F16:
        self_attention_kernel<llaisys::fp16_t>((llaisys::fp16_t *)attn_val, (const llaisys::fp16_t *)q, 
                                               (const llaisys::fp16_t *)k, (const llaisys::fp16_t *)v,
                                               seqlen, total_len, nhead, nkvhead, d, dv, scale);
        break;
    case LLAISYS_DTYPE_BF16:
        self_attention_kernel<llaisys::bf16_t>((llaisys::bf16_t *)attn_val, (const llaisys::bf16_t *)q, 
                                               (const llaisys::bf16_t *)k, (const llaisys::bf16_t *)v,
                                               seqlen, total_len, nhead, nkvhead, d, dv, scale);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu