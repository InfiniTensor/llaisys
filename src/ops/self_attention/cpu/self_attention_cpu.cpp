#include "self_attention_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <vector>
#include <limits>

template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v, 
                     size_t qlen, size_t kvlen, size_t nhead, size_t nkvhead, size_t d, float scale) {
    size_t head_repeat = nhead / nkvhead;
    
    for (size_t qi = 0; qi < qlen; qi++) {
        for (size_t h = 0; h < nhead; h++) {
            size_t kv_h = h / head_repeat;
            std::vector<float> scores(kvlen);
            
            // Compute Q * K^T * scale
            for (size_t ki = 0; ki < kvlen; ki++) {
                float sum = 0.0f;
                for (size_t j = 0; j < d; j++) {
                    float q_val = llaisys::utils::cast<float>(q[qi * nhead * d + h * d + j]);
                    float k_val = llaisys::utils::cast<float>(k[ki * nkvhead * d + kv_h * d + j]);
                    sum += q_val * k_val;
                }
                scores[ki] = sum * scale;
            }
            
            // Apply causal mask and softmax
            float max_score = -std::numeric_limits<float>::infinity();
            for (size_t ki = 0; ki < kvlen; ki++) {
                if (ki <= kvlen - qlen + qi) {  // causal mask
                    max_score = std::max(max_score, scores[ki]);
                }
            }
            
            float sum_exp = 0.0f;
            for (size_t ki = 0; ki < kvlen; ki++) {
                if (ki <= kvlen - qlen + qi) {
                    scores[ki] = std::exp(scores[ki] - max_score);
                    sum_exp += scores[ki];
                } else {
                    scores[ki] = 0.0f;
                }
            }
            
            for (size_t ki = 0; ki < kvlen; ki++) {
                scores[ki] /= sum_exp;
            }
            
            // Compute attention * V
            for (size_t j = 0; j < d; j++) {
                float sum = 0.0f;
                for (size_t ki = 0; ki < kvlen; ki++) {
                    float v_val = llaisys::utils::cast<float>(v[ki * nkvhead * d + kv_h * d + j]);
                    sum += scores[ki] * v_val;
                }
                attn_val[qi * nhead * d + h * d + j] = llaisys::utils::cast<T>(sum);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t type, size_t qlen, size_t kvlen, size_t nhead, size_t nkvhead, size_t d, float scale) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val), reinterpret_cast<const float *>(q),
                              reinterpret_cast<const float *>(k), reinterpret_cast<const float *>(v),
                              qlen, kvlen, nhead, nkvhead, d, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val), reinterpret_cast<const llaisys::bf16_t *>(q),
                              reinterpret_cast<const llaisys::bf16_t *>(k), reinterpret_cast<const llaisys::bf16_t *>(v),
                              qlen, kvlen, nhead, nkvhead, d, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val), reinterpret_cast<const llaisys::fp16_t *>(q),
                              reinterpret_cast<const llaisys::fp16_t *>(k), reinterpret_cast<const llaisys::fp16_t *>(v),
                              qlen, kvlen, nhead, nkvhead, d, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
}
