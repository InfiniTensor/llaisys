#include "self_attention_cpu.hpp"
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include "../../../utils.hpp"

template <typename T>
static void self_attention_(T *out, const T *q, const T *k, const T *v, 
    float scale,size_t qlen, size_t kvlen, 
    size_t nh, size_t nkvh, size_t hd, size_t dv) {
    //cal group size of kv head
    size_t kv_group = nh / nkvh;
    // attn scores
    std::vector<float> scores(kvlen);

    // for each q head
    for (size_t h = 0; h < nh; ++h) {
        size_t kvh = h / kv_group;
        for (size_t i = 0; i < qlen; ++i) {
            const T *q_ = q + (i * nh + h) * hd;
            float max_score = -std::numeric_limits<float>::infinity();
            // causal mask
            int64_t limit = static_cast<int64_t>(i) + 
            static_cast<int64_t>(kvlen) - static_cast<int64_t>(qlen);

            for (size_t j = 0; j < kvlen; ++j) {
                // for j > i, set score to -inf, because it can't be seen
                if (static_cast<int64_t>(j) > limit) {
                    scores[j] = -std::numeric_limits<float>::infinity();
                    continue;
                }
                // start to cal attn scores
                // QK^T
                float acc = 0.0f;
                const T *k_ = k + (j * nkvh + kvh) * hd;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> 
                    || std::is_same_v<T, llaisys::fp16_t>) {
                    for (size_t d = 0; d < hd; ++d) {
                        acc += llaisys::utils::cast<float>(q_[d]) 
                        * llaisys::utils::cast<float>(k_[d]);
                    }
                } else {
                    for (size_t d = 0; d < hd; ++d) {
                        acc += q_[d] * k_[d];
                    }
                }
                //QK^T * scale
                acc *= scale;
                scores[j] = acc;
                if (acc > max_score) {
                    max_score = acc;
                }
            }

            float sum = 0.0f;
            for (size_t j = 0; j < kvlen; ++j) {
                if (scores[j] == -std::numeric_limits<float>::infinity()) {
                    scores[j] = 0.0f;
                    continue;
                }
                // softmax exp(j - max_score)
                float w = std::exp(scores[j] - max_score);
                scores[j] = w;
                sum += w;
            }
            // softmax sum(exp(j - max_score))
            float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
            
            // start to cal causalsoftmax(A)*V
            T *out_ = out + (i * nh + h) * dv;
            for (size_t d = 0; d < dv; ++d) {
                float acc = 0.0f;
                for (size_t j = 0; j < kvlen; ++j) {
                    if (scores[j] == 0.0f) {
                        continue;
                    }
                    const T *v_ = v + (j * nkvh + kvh) * dv;
                    if constexpr (std::is_same_v<T, llaisys::bf16_t> 
                        || std::is_same_v<T, llaisys::fp16_t>) {
                        acc += scores[j] * llaisys::utils::cast<float>(v_[d]);
                    } else {
                        acc += scores[j] * v_[d];
                    }
                }
                if constexpr (std::is_same_v<T, llaisys::bf16_t> 
                    || std::is_same_v<T, llaisys::fp16_t>) {
                    out_[d] = llaisys::utils::cast<T>(acc * inv_sum);
                } else {
                    out_[d] = static_cast<T>(acc * inv_sum);
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *out, const std::byte *q, const std::byte *k, 
    const std::byte *v, float scale,
    llaisysDataType_t type, size_t qlen, 
    size_t kvlen, size_t nh, size_t nkvh, size_t hd, size_t dv) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_<float>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(q),
            reinterpret_cast<const float *>(k),
            reinterpret_cast<const float *>(v),
            scale, qlen, kvlen, nh, nkvh, hd, dv);
    case LLAISYS_DTYPE_BF16:
        return self_attention_<llaisys::bf16_t>(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(q),
            reinterpret_cast<const llaisys::bf16_t *>(k),
            reinterpret_cast<const llaisys::bf16_t *>(v),
            scale, qlen, kvlen, nh, nkvh, hd, dv);
    case LLAISYS_DTYPE_F16:
        return self_attention_<llaisys::fp16_t>(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(q),
            reinterpret_cast<const llaisys::fp16_t *>(k),
            reinterpret_cast<const llaisys::fp16_t *>(v),
            scale, qlen, kvlen, nh, nkvh, hd, dv);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
}