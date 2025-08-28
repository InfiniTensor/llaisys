#include "self_attention_cpu.hpp"
#include "../../../utils.hpp"

#include <cmath>
#include <vector>

namespace {

template <typename T>
inline void self_attn_impl(T *out,
                           const T *q,
                           const T *k,
                           const T *v,
                           size_t seqlen, size_t nhead, size_t d,
                           size_t total_len, size_t nkvhead, size_t dv,
                           float scale) {
    const size_t group = nhead / nkvhead; // GQA 
    const size_t prefix = (total_len >= seqlen) ? (total_len - seqlen) : 0;

    std::vector<float> score(total_len);

    for (size_t t = 0; t < seqlen; ++t) {
        const size_t allowed = prefix + t + 1; 
        for (size_t h = 0; h < nhead; ++h) {
            const size_t kvh = h / group;

            const T *q_vec = q + (t * nhead + h) * d;
            T *y_vec = out + (t * nhead + h) * dv;

            // QK^T * scale
            float max_s = -INFINITY;
            for (size_t j = 0; j < allowed; ++j) {
                const T *k_vec = k + (j * nkvhead + kvh) * d;

                float acc = 0.f;
                for (size_t u = 0; u < d; ++u) {
                    float qf = llaisys::utils::cast<float>(q_vec[u]);
                    float kf = llaisys::utils::cast<float>(k_vec[u]);
                    acc += qf * kf;
                }
                acc *= scale;
                score[j] = acc;
                if (acc > max_s) {
                    max_s = acc;
                }
            }

            // 2)softmax
            float denom = 0.f;
            for (size_t j = 0; j < allowed; ++j) {
                float e = std::exp(score[j] - max_s);
                score[j] = e; //  e^{...}
                denom += e;
            }
            const float inv_denom = 1.f / denom;

            // 3) 
            for (size_t c = 0; c < dv; ++c) {
                float acc = 0.f;
                for (size_t j = 0; j < allowed; ++j) {
                    const T *v_vec = v + (j * nkvhead + kvh) * dv;
                    acc += score[j] * llaisys::utils::cast<float>(v_vec[c]);
                }
                y_vec[c] = llaisys::utils::cast<T>(acc * inv_denom);
            }
        }
    }
}

} // anonymous namespace

namespace llaisys::ops::cpu {

void self_attention(std::byte *out,
                    const std::byte *q,
                    const std::byte *k,
                    const std::byte *v,
                    llaisysDataType_t type,
                    size_t seqlen, size_t nhead, size_t d,
                    size_t total_len, size_t nkvhead, size_t dv,
                    float scale) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attn_impl(reinterpret_cast<float *>(out),
                              reinterpret_cast<const float *>(q),
                              reinterpret_cast<const float *>(k),
                              reinterpret_cast<const float *>(v),
                              seqlen, nhead, d, total_len, nkvhead, dv, scale);
    case LLAISYS_DTYPE_F16:
        return self_attn_impl(reinterpret_cast<llaisys::fp16_t *>(out),
                              reinterpret_cast<const llaisys::fp16_t *>(q),
                              reinterpret_cast<const llaisys::fp16_t *>(k),
                              reinterpret_cast<const llaisys::fp16_t *>(v),
                              seqlen, nhead, d, total_len, nkvhead, dv, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attn_impl(reinterpret_cast<llaisys::bf16_t *>(out),
                              reinterpret_cast<const llaisys::bf16_t *>(q),
                              reinterpret_cast<const llaisys::bf16_t *>(k),
                              reinterpret_cast<const llaisys::bf16_t *>(v),
                              seqlen, nhead, d, total_len, nkvhead, dv, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu
