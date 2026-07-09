#include "self_attention_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <vector>
#include <algorithm>

template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v,
                     size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead, 
                     size_t d, size_t dv, float scale) {
    size_t n_groups = nhead / nkvhead; // GQA 分组大小

    for (size_t h = 0; h < nhead; ++h) {
        size_t h_kv = h / n_groups; // 对应的 KV 头索引
        
        for (size_t i = 0; i < seqlen; ++i) {
            std::vector<float> scores(total_len);
            float max_score = -INFINITY;

            // 1. 计算 QK^T * scale 并应用因果掩码
            size_t q_idx_base = (i * nhead + h) * d;
            for (size_t j = 0; j < total_len; ++j) {
                // 因果掩码：当前位置 i 只能看到过去的位置 j
                if (j > (total_len - seqlen + i)) {
                    scores[j] = -INFINITY;
                    continue;
                }

                float dot = 0.0f;
                size_t k_idx_base = (j * nkvhead + h_kv) * d;
                for (size_t c = 0; c < d; ++c) {
                    dot += llaisys::utils::cast<float>(q[q_idx_base + c]) * llaisys::utils::cast<float>(k[k_idx_base + c]);
                }
                scores[j] = dot * scale;
                max_score = std::max(max_score, scores[j]);
            }

            // 2. Softmax 归一化
            float sum_exp = 0.0f;
            for (size_t j = 0; j < total_len; ++j) {
                if (scores[j] != -INFINITY) {
                    scores[j] = std::exp(scores[j] - max_score);
                    sum_exp += scores[j];
                } else {
                    scores[j] = 0.0f;
                }
            }
            for (size_t j = 0; j < total_len; ++j) scores[j] /= sum_exp;

            // 3. 计算 Softmax(A) * V
            size_t out_idx_base = (i * nhead + h) * dv;
            for (size_t cdv = 0; cdv < dv; ++cdv) {
                float res = 0.0f;
                for (size_t j = 0; j < total_len; ++j) {
                    if (scores[j] > 0.0f) {
                        size_t v_idx = (j * nkvhead + h_kv) * dv + cdv;
                        res += scores[j] * llaisys::utils::cast<float>(v[v_idx]);
                    }
                }
                attn_val[out_idx_base + cdv] = llaisys::utils::cast<T>(res);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t type, size_t seqlen, size_t total_len, 
                    size_t nhead, size_t nkvhead, size_t d, size_t dv, float scale) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val), reinterpret_cast<const float *>(q),
                                reinterpret_cast<const float *>(k), reinterpret_cast<const float *>(v),
                                seqlen, total_len, nhead, nkvhead, d, dv, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val), reinterpret_cast<const llaisys::bf16_t *>(q),
                                reinterpret_cast<const llaisys::bf16_t *>(k), reinterpret_cast<const llaisys::bf16_t *>(v),
                                seqlen, total_len, nhead, nkvhead, d, dv, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val), reinterpret_cast<const llaisys::fp16_t *>(q),
                                reinterpret_cast<const llaisys::fp16_t *>(k), reinterpret_cast<const llaisys::fp16_t *>(v),
                                seqlen, total_len, nhead, nkvhead, d, dv, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
}