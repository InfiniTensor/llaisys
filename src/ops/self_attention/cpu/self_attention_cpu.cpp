#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <vector>
#include <limits>
#include <algorithm>

template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v,
                     size_t seqlen, size_t total_len,
                     size_t nhead, size_t nkvhead,
                     size_t d, size_t dv,
                     float scale) {
    

    size_t n_rep = nhead / nkvhead;

    std::vector<float> scores(total_len);

    for (size_t i = 0; i < seqlen; ++i) {

        size_t current_pos = (total_len - seqlen) + i;

        // Loop over Heads
        for (size_t h = 0; h < nhead; ++h) {
            

            size_t kv_h = h / n_rep;

            // Q shape: [seqlen, nhead, d]
            const T* q_vec = q + (i * nhead * d) + (h * d);

            float max_score = -std::numeric_limits<float>::infinity();

            for (size_t t = 0; t < total_len; ++t) {
                if (t > current_pos) {
                    scores[t] = -std::numeric_limits<float>::infinity();
                    continue;
                }

                const T* k_vec = k + (t * nkvhead * d) + (kv_h * d);

                float dot = 0.0f;
                for (size_t idx = 0; idx < d; ++idx) {
                    float q_val, k_val;
                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        q_val = llaisys::utils::cast<float>(q_vec[idx]);
                        k_val = llaisys::utils::cast<float>(k_vec[idx]);
                    } else {
                        q_val = static_cast<float>(q_vec[idx]);
                        k_val = static_cast<float>(k_vec[idx]);
                    }
                    dot += q_val * k_val;
                }

                float score = dot * scale;
                scores[t] = score;
                if (score > max_score) {
                    max_score = score;
                }
            }

            float sum_exp = 0.0f;
            for (size_t t = 0; t <= current_pos; ++t) {
                float exp_val = std::exp(scores[t] - max_score);
                scores[t] = exp_val;
                sum_exp += exp_val;
            }


            T* out_vec = attn_val + (i * nhead * dv) + (h * dv);


            std::vector<float> acc(dv, 0.0f);

            for (size_t t = 0; t <= current_pos; ++t) {
                float prob = scores[t] / sum_exp;
                
                // Get V vector at time t, head kv_h
                const T* v_vec = v + (t * nkvhead * dv) + (kv_h * dv);

                for (size_t j = 0; j < dv; ++j) {
                    float v_val;
                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        v_val = llaisys::utils::cast<float>(v_vec[j]);
                    } else {
                        v_val = static_cast<float>(v_vec[j]);
                    }
                    acc[j] += prob * v_val;
                }
            }

            for (size_t j = 0; j < dv; ++j) {
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    out_vec[j] = llaisys::utils::cast<T>(acc[j]);
                } else {
                    out_vec[j] = static_cast<T>(acc[j]);
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, 
                    llaisysDataType_t type, 
                    size_t seqlen, size_t total_len, 
                    size_t nhead, size_t nkvhead, 
                    size_t d, size_t dv, 
                    float scale) {
    
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val), 
                               reinterpret_cast<const float *>(q),
                               reinterpret_cast<const float *>(k),
                               reinterpret_cast<const float *>(v),
                               seqlen, total_len, nhead, nkvhead, d, dv, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val), 
                               reinterpret_cast<const llaisys::bf16_t *>(q),
                               reinterpret_cast<const llaisys::bf16_t *>(k),
                               reinterpret_cast<const llaisys::bf16_t *>(v),
                               seqlen, total_len, nhead, nkvhead, d, dv, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val), 
                               reinterpret_cast<const llaisys::fp16_t *>(q),
                               reinterpret_cast<const llaisys::fp16_t *>(k),
                               reinterpret_cast<const llaisys::fp16_t *>(v),
                               seqlen, total_len, nhead, nkvhead, d, dv, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
}