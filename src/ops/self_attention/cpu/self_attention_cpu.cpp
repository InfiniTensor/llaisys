#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

template <typename T>
static void self_attention_(T *out,
                            const T *q,
                            const T *k,
                            const T *v,
                            size_t qlen,
                            size_t nh,
                            size_t kvlen,
                            size_t nkvh,
                            size_t hd,
                            float scale) {
    // GQA mapping: repeat each kv head nh/nkvh times (same as repeat_interleave in the test).
    const size_t repeat = nh / nkvh;
    const ptrdiff_t offset = static_cast<ptrdiff_t>(kvlen) - static_cast<ptrdiff_t>(qlen);

    std::vector<float> logits(kvlen);
    std::vector<float> probs(kvlen);

    for (size_t h = 0; h < nh; h++) {
        const size_t hk = h / repeat;

        for (size_t qi = 0; qi < qlen; qi++) {
            const ptrdiff_t max_j = std::min<ptrdiff_t>(
                static_cast<ptrdiff_t>(kvlen) - 1,
                static_cast<ptrdiff_t>(qi) + offset);

            float max_logit = -std::numeric_limits<float>::infinity();

            // 1) logits = (q_i · k_j) * scale, with causal mask.
            for (size_t kj = 0; kj < kvlen; kj++) {
                if (static_cast<ptrdiff_t>(kj) > max_j) {
                    logits[kj] = -std::numeric_limits<float>::infinity();
                    continue;
                }

                const T *q_row = q + (qi * nh + h) * hd;
                const T *k_row = k + (kj * nkvh + hk) * hd;
                float dot = 0.0f;
                for (size_t d = 0; d < hd; d++) {
                    dot += llaisys::utils::cast<float>(q_row[d]) * llaisys::utils::cast<float>(k_row[d]);
                }
                const float logit = dot * scale;
                logits[kj] = logit;
                max_logit = std::max(max_logit, logit);
            }

            // 2) probs = softmax(logits)
            float denom = 0.0f;
            for (size_t kj = 0; kj < kvlen; kj++) {
                float p = 0.0f;
                if (!std::isinf(logits[kj])) {
                    p = std::exp(logits[kj] - max_logit);
                }
                probs[kj] = p;
                denom += p;
            }
            const float inv_denom = 1.0f / denom;

            // 3) out = probs @ v
            T *out_row = out + (qi * nh + h) * hd;
            for (size_t d = 0; d < hd; d++) {
                float acc = 0.0f;
                for (size_t kj = 0; kj < kvlen; kj++) {
                    if (probs[kj] == 0.0f) continue;
                    const T *v_row = v + (kj * nkvh + hk) * hd;
                    acc += (probs[kj] * inv_denom) * llaisys::utils::cast<float>(v_row[d]);
                }
                out_row[d] = llaisys::utils::cast<T>(acc);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *out,
                    const std::byte *q,
                    const std::byte *k,
                    const std::byte *v,
                    llaisysDataType_t dtype,
                    size_t qlen,
                    size_t nh,
                    size_t kvlen,
                    size_t nkvh,
                    size_t hd,
                    float scale) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(out),
                               reinterpret_cast<const float *>(q),
                               reinterpret_cast<const float *>(k),
                               reinterpret_cast<const float *>(v),
                               qlen, nh, kvlen, nkvh, hd, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(out),
                               reinterpret_cast<const llaisys::bf16_t *>(q),
                               reinterpret_cast<const llaisys::bf16_t *>(k),
                               reinterpret_cast<const llaisys::bf16_t *>(v),
                               qlen, nh, kvlen, nkvh, hd, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(out),
                               reinterpret_cast<const llaisys::fp16_t *>(q),
                               reinterpret_cast<const llaisys::fp16_t *>(k),
                               reinterpret_cast<const llaisys::fp16_t *>(v),
                               qlen, nh, kvlen, nkvh, hd, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
