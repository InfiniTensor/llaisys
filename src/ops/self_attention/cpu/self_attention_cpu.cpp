#include "self_attention_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <vector>
#include <algorithm> 
#include <limits>    

namespace { 

template <typename T>
void self_attention_(T *out, const T *q, const T *k, const T *v,
                     size_t seqlen, size_t total_len, 
                     size_t nhead, size_t nkvhead, 
                     size_t head_dim, size_t head_dim_v,
                     float scale) {
    

    size_t group_size = nhead / nkvhead;

    std::vector<float> scores(total_len);

    size_t start_pos = total_len - seqlen;

    for (size_t i = 0; i < seqlen; ++i) {
        size_t q_global_pos = start_pos + i; 

        for (size_t h = 0; h < nhead; ++h) {
            size_t kv_h = h / group_size;

            float max_score = -std::numeric_limits<float>::infinity();

            const T* q_vec = q + (i * nhead * head_dim) + (h * head_dim);

            for (size_t t = 0; t < total_len; ++t) {
                if (t > q_global_pos) {
                    scores[t] = -std::numeric_limits<float>::infinity();
                    continue;
                }

                const T* k_vec = k + (t * nkvhead * head_dim) + (kv_h * head_dim);

                float dot = 0.0f;
                for (size_t d = 0; d < head_dim; ++d) {
                    dot += llaisys::utils::cast<float>(q_vec[d]) * llaisys::utils::cast<float>(k_vec[d]);
                }
                dot *= scale;
                scores[t] = dot;

                if (dot > max_score) {
                    max_score = dot;
                }
            }

            float sum_exp = 0.0f;
            for (size_t t = 0; t < total_len; ++t) {
                if (scores[t] == -std::numeric_limits<float>::infinity()) {
                    scores[t] = 0.0f; // exp(-inf) = 0
                } else {
                    float val = std::exp(scores[t] - max_score);
                    scores[t] = val;
                    sum_exp += val;
                }
            }

            float inv_sum = 1.0f / (sum_exp + 1e-6f); 
            for (size_t t = 0; t < total_len; ++t) {
                scores[t] *= inv_sum;
            }

            T* out_vec = out + (i * nhead * head_dim_v) + (h * head_dim_v);

            std::vector<float> acc_out(head_dim_v, 0.0f);

            for (size_t t = 0; t < total_len; ++t) {
                float weight = scores[t];
                if (weight == 0.0f) continue; 

                const T* v_vec = v + (t * nkvhead * head_dim_v) + (kv_h * head_dim_v);

                for (size_t d = 0; d < head_dim_v; ++d) {
                    acc_out[d] += weight * llaisys::utils::cast<float>(v_vec[d]);
                }
            }

            for (size_t d = 0; d < head_dim_v; ++d) {
                out_vec[d] = llaisys::utils::cast<T>(acc_out[d]);
            }
        }
    }
}

} // namespace

namespace llaisys::ops::cpu {

void self_attention(std::byte *out, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t dtype, 
                    size_t seqlen, size_t total_len, 
                    size_t nhead, size_t nkvhead, 
                    size_t head_dim, size_t head_dim_v,
                    float scale) {
    switch (dtype) {
        case LLAISYS_DTYPE_F32:
            return self_attention_(reinterpret_cast<float*>(out),
                                   reinterpret_cast<const float*>(q),
                                   reinterpret_cast<const float*>(k),
                                   reinterpret_cast<const float*>(v),
                                   seqlen, total_len, nhead, nkvhead, head_dim, head_dim_v, scale);
        case LLAISYS_DTYPE_F16:
            return self_attention_(reinterpret_cast<llaisys::fp16_t*>(out),
                                   reinterpret_cast<const llaisys::fp16_t*>(q),
                                   reinterpret_cast<const llaisys::fp16_t*>(k),
                                   reinterpret_cast<const llaisys::fp16_t*>(v),
                                   seqlen, total_len, nhead, nkvhead, head_dim, head_dim_v, scale);
        case LLAISYS_DTYPE_BF16:
            return self_attention_(reinterpret_cast<llaisys::bf16_t*>(out),
                                   reinterpret_cast<const llaisys::bf16_t*>(q),
                                   reinterpret_cast<const llaisys::bf16_t*>(k),
                                   reinterpret_cast<const llaisys::bf16_t*>(v),
                                   seqlen, total_len, nhead, nkvhead, head_dim, head_dim_v, scale);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu