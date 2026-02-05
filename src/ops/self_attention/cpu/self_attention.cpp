#include "self_attention.hpp"

#include "../../../utils.hpp"

#include <vector>
#include <algorithm>
#include <cmath>

template <typename T>
void self_attention_(
    T* out, const T* q, const T* k, const T* v,
    size_t seqlen, size_t nhead, size_t d,
    size_t total_len, size_t nkvhead, size_t dv,
    float scale
) {
    using namespace llaisys::utils;
    for (size_t h = 0; h < nhead; h++) {
        size_t q_base = h * d;
        size_t k_base = h / (nhead / nkvhead) * d;
        size_t v_base = h / (nhead / nkvhead) * dv;

        for (size_t i = 0; i < seqlen; i++) {
            size_t len = total_len - seqlen + 1;
            std::vector<float> attn_weights(len + i);
            for (size_t j = 0; j < i + len; j++) {
                float sum = 0.f;
                for (size_t n = 0; n < d; n++) {
                    sum += cast<float>(
                        q[q_base + i * nhead * d + n] * k[k_base  + j * nkvhead * d + n]
                    );
                }
                sum *= scale;
                attn_weights[j] = sum;
            }
            // softmax
            float max_weight = *std::max_element(attn_weights.begin(), attn_weights.end());
            float sum_exp = 0.f;
            for (float& w : attn_weights) {
                w = std::exp(w - max_weight);
                sum_exp += w;
            }
            for (float& w : attn_weights) {
                w /= sum_exp;
            }
            // output
            for (size_t j = 0; j < dv; j++) {
                float sum = 0.f;
                for (size_t n = 0; n < attn_weights.size(); n++) {
                    sum += attn_weights[n] * cast<float>(v[v_base + n * nkvhead * dv + j]);
                }
                out[(h + i * nhead) * dv + j] = cast<T>(sum);
            }
        }
   }
}

namespace llaisys::ops::cpu {
void self_attention(
    void* out, const void* q, const void* k, const void* v,
    size_t seqlen, size_t nhead, size_t d,
    size_t total_len, size_t nkvhead, size_t dv,
    float scale,
    llaisysDataType_t dtype
) {
    switch (dtype) {
        case LLAISYS_DTYPE_F32:
            return self_attention_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(q),
                                   reinterpret_cast<const float *>(k), reinterpret_cast<const float *>(v),
                                   seqlen, nhead, d, total_len, nkvhead, dv, scale);
        case LLAISYS_DTYPE_BF16:
            return self_attention_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(q),
                                   reinterpret_cast<const llaisys::bf16_t *>(k), reinterpret_cast<const llaisys::bf16_t *>(v),
                                   seqlen, nhead, d, total_len, nkvhead, dv, scale);
        case LLAISYS_DTYPE_F16:
            return self_attention_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(q),
                                   reinterpret_cast<const llaisys::fp16_t *>(k), reinterpret_cast<const llaisys::fp16_t *>(v),
                                   seqlen, nhead, d, total_len, nkvhead, dv, scale);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
} // namespace llaisys::ops::cpu
}