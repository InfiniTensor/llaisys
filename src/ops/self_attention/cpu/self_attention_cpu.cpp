#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <algorithm>
#include <limits>

// Self-Attention 实现
// q: [qlen, nh, hd]
// k: [kvlen, nkvh, hd] 
// v: [kvlen, nkvh, hd]
// attn_val: [qlen, nh, hd]
template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v,
                    size_t qlen, size_t kvlen, size_t nh, size_t nkvh, size_t hd, float scale) {
    
    // Group Query Attention: 每个kv头对应多少个query头
    size_t group_size = nh / nkvh;
    
    // 为每个query位置和每个头计算注意力
    for (size_t qi = 0; qi < qlen; qi++) {
        for (size_t h = 0; h < nh; h++) {
            // 确定当前query头对应的kv头
            size_t kv_head = h / group_size;
            
            const T *q_head = q + (qi * nh + h) * hd;  // 当前query向量
            T *out_head = attn_val + (qi * nh + h) * hd;  // 输出位置
            
            // 步骤1：计算注意力分数 Q @ K^T * scale
            float *attn_scores = new float[kvlen];  // 注意力分数
            
            for (size_t ki = 0; ki < kvlen; ki++) {
                const T *k_head = k + (ki * nkvh + kv_head) * hd;  // 对应的key向量
                
                // 计算点积 q · k
                float score = 0.0f;
                for (size_t d = 0; d < hd; d++) {
                    float q_val, k_val;
                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        q_val = llaisys::utils::cast<float>(q_head[d]);
                        k_val = llaisys::utils::cast<float>(k_head[d]);
                    } else {
                        q_val = static_cast<float>(q_head[d]);
                        k_val = static_cast<float>(k_head[d]);
                    }
                    score += q_val * k_val;
                }
                
                score *= scale;  // 应用缩放因子
                
                // 步骤2：应用因果mask
                // PyTorch逻辑: mask = ~torch.ones(L, S).tril(diagonal=S-L) 
                // 即：mask[i,j] = True if j > i + (S-L)
                if (static_cast<int>(ki) > static_cast<int>(qi) + static_cast<int>(kvlen) - static_cast<int>(qlen)) {
                    score = -std::numeric_limits<float>::infinity();
                }
                
                attn_scores[ki] = score;
            }
            
            // 步骤3：Softmax归一化
            // 先找到最大值（数值稳定性）
            float max_score = -std::numeric_limits<float>::infinity();
            for (size_t ki = 0; ki < kvlen; ki++) {
                if (attn_scores[ki] > max_score) {
                    max_score = attn_scores[ki];
                }
            }
            
            // 计算exp和sum
            float sum_exp = 0.0f;
            for (size_t ki = 0; ki < kvlen; ki++) {
                attn_scores[ki] = std::exp(attn_scores[ki] - max_score);
                sum_exp += attn_scores[ki];
            }
            
            // 归一化
            for (size_t ki = 0; ki < kvlen; ki++) {
                attn_scores[ki] /= sum_exp;
            }
            
            // 步骤4：加权求和 attn_scores @ V
            for (size_t d = 0; d < hd; d++) {
                float output_val = 0.0f;
                
                for (size_t ki = 0; ki < kvlen; ki++) {
                    const T *v_head = v + (ki * nkvh + kv_head) * hd;  // 对应的value向量
                    
                    float v_val;
                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        v_val = llaisys::utils::cast<float>(v_head[d]);
                    } else {
                        v_val = static_cast<float>(v_head[d]);
                    }
                    
                    output_val += attn_scores[ki] * v_val;
                }
                
                // 存储结果
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    out_head[d] = llaisys::utils::cast<T>(output_val);
                } else {
                    out_head[d] = static_cast<T>(output_val);
                }
            }
            
            delete[] attn_scores;
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                   llaisysDataType_t type, size_t qlen, size_t kvlen, size_t nh, size_t nkvh, 
                   size_t hd, float scale) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(
            reinterpret_cast<float *>(attn_val),
            reinterpret_cast<const float *>(q),
            reinterpret_cast<const float *>(k),
            reinterpret_cast<const float *>(v),
            qlen, kvlen, nh, nkvh, hd, scale
        );
    case LLAISYS_DTYPE_BF16:
        return self_attention_(
            reinterpret_cast<llaisys::bf16_t *>(attn_val),
            reinterpret_cast<const llaisys::bf16_t *>(q),
            reinterpret_cast<const llaisys::bf16_t *>(k),
            reinterpret_cast<const llaisys::bf16_t *>(v),
            qlen, kvlen, nh, nkvh, hd, scale
        );
    case LLAISYS_DTYPE_F16:
        return self_attention_(
            reinterpret_cast<llaisys::fp16_t *>(attn_val),
            reinterpret_cast<const llaisys::fp16_t *>(q),
            reinterpret_cast<const llaisys::fp16_t *>(k),
            reinterpret_cast<const llaisys::fp16_t *>(v),
            qlen, kvlen, nh, nkvh, hd, scale
        );
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
