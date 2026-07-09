#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids,
           const size_t seqlen, const size_t nhead, const size_t d, float theta) {
    size_t half = d / 2;
    std::vector<float> denom(half);
    for (size_t j = 0; j < half; ++j) {
        float exp = (2.0f * j) / d;        // 2j/d
        denom[j] = std::pow(theta, exp); // θ^(2j/d)
    }

    std::vector<float> sin_cache(seqlen * half);
    std::vector<float> cos_cache(seqlen * half);
    for (size_t i = 0; i < seqlen; ++i) {
        float pos = static_cast<float>(pos_ids[i]);
        for (size_t j = 0; j < half; ++j) {
            float phi = pos / denom[j];
            sin_cache[i * half + j] = sinf(phi);
            cos_cache[i * half + j] = cosf(phi);
        }
    }

    for (size_t i = 0; i < seqlen; ++i) {
        for (size_t k = 0; k < nhead; ++k) {
            size_t base = i * (nhead * d) + k * d;
            for (size_t j = 0; j < half; ++j) {
                float cos_v = cos_cache[i * half + j];
                float sin_v = sin_cache[i * half + j];
                if constexpr ((std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>)) {
                    out[base + j]
                        = llaisys::utils::cast<T>(llaisys::utils::cast<float>(in[base + j]) * cos_v
                                                  - llaisys::utils::cast<float>(in[base + j + half]) * sin_v);
                    out[base + j + half]
                        = llaisys::utils::cast<T>(llaisys::utils::cast<float>(in[base + j + half]) * cos_v
                                                  + llaisys::utils::cast<float>(in[base + j]) * sin_v);
                } else {
                    out[base + j] = in[base + j] * cos_v
                                  - in[base + j + half] * sin_v;
                    out[base + j + half] = in[base + j + half] * cos_v
                                         + in[base + j] * sin_v;
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          llaisysDataType_t type, const size_t seqlen, const size_t nhead, const size_t d, float theta) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                     reinterpret_cast<const int64_t *>(pos_ids), seqlen, nhead, d, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                     reinterpret_cast<const int64_t *>(pos_ids), seqlen, nhead, d, theta);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                     reinterpret_cast<const int64_t *>(pos_ids), seqlen, nhead, d, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
