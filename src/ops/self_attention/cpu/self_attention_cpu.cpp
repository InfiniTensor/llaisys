#include "self_attention_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace llaisys::ops::cpu {

template <typename T>
void self_attention_t(std::byte* out_raw,
                      const std::byte* q_raw,
                      const std::byte* k_raw,
                      const std::byte* v_raw,
                      size_t qlen,
                      size_t kvlen,
                      size_t nhead,
                      size_t nkvhead,
                      size_t head_dim,
                      size_t value_dim,
                      float scale) {
    const T* q = reinterpret_cast<const T*>(q_raw);
    const T* k = reinterpret_cast<const T*>(k_raw);
    const T* v = reinterpret_cast<const T*>(v_raw);
    T* out = reinterpret_cast<T*>(out_raw);

    const size_t head_repeat = nhead / nkvhead;
    const size_t causal_offset = kvlen - qlen;
    std::vector<float> scores(kvlen);
    std::vector<float> weights(kvlen);

    for (size_t i = 0; i < qlen; ++i) {
        for (size_t h = 0; h < nhead; ++h) {
            size_t kvh = h / head_repeat;
            const T* q_vec = q + (i * nhead + h) * head_dim;

            float max_score = -std::numeric_limits<float>::infinity();
            for (size_t j = 0; j < kvlen; ++j) {
                if (j > i + causal_offset) {
                    scores[j] = -std::numeric_limits<float>::infinity();
                    continue;
                }
                const T* k_vec = k + (j * nkvhead + kvh) * head_dim;
                float dot = 0.0f;
                for (size_t d = 0; d < head_dim; ++d) {
                    dot += llaisys::utils::cast<float>(q_vec[d]) *
                           llaisys::utils::cast<float>(k_vec[d]);
                }
                float s = dot * scale;
                scores[j] = s;
                if (s > max_score) max_score = s;
            }

            float denom = 0.0f;
            for (size_t j = 0; j < kvlen; ++j) {
                if (scores[j] == -std::numeric_limits<float>::infinity()) {
                    weights[j] = 0.0f;
                    continue;
                }
                float w = std::exp(scores[j] - max_score);
                weights[j] = w;
                denom += w;
            }
            float inv_denom = denom > 0.0f ? 1.0f / denom : 0.0f;
            for (size_t j = 0; j < kvlen; ++j) {
                weights[j] *= inv_denom; 
            }

            T* out_vec = out + (i * nhead + h) * value_dim;
            for (size_t d = 0; d < value_dim; ++d) {
                float acc = 0.0f;
                for (size_t j = 0; j < kvlen; ++j) {
                    if (weights[j] == 0.0f) continue;
                    const T* v_vec = v + (j * nkvhead + kvh) * value_dim;
                    acc += weights[j] * llaisys::utils::cast<float>(v_vec[d]);
                }
                out_vec[d] = llaisys::utils::cast<T>(acc);
            }
        }
    }
}

void self_attention(std::byte* out,
                    const std::byte* q,
                    const std::byte* k,
                    const std::byte* v,
                    llaisysDataType_t dtype,
                    size_t qlen,
                    size_t kvlen,
                    size_t nhead,
                    size_t nkvhead,
                    size_t head_dim,
                    size_t value_dim,
                    float scale) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return self_attention_t<float>(out, q, k, v, qlen, kvlen, nhead, nkvhead, head_dim, value_dim, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_t<llaisys::fp16_t>(out, q, k, v, qlen, kvlen, nhead, nkvhead, head_dim, value_dim, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_t<llaisys::bf16_t>(out, q, k, v, qlen, kvlen, nhead, nkvhead, head_dim, value_dim, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu
