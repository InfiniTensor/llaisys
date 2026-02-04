#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rope_(std::byte *out_raw, const std::byte *in_raw, const std::byte *pos_ids_raw, size_t seqlen, size_t nhead, size_t d, float theta) {
    T *out = reinterpret_cast<T*>(out_raw);
    const T *in = reinterpret_cast<const T*>(in_raw);
    const int64_t *pos_ids = reinterpret_cast<const int64_t*>(pos_ids_raw);

    const size_t half_d = d / 2;

    for (size_t i = 0; i < seqlen; ++i) {
        float p = static_cast<float>(pos_ids[i]);
        for (size_t h = 0; h < nhead; ++h) {
            for (size_t j = 0; j < half_d; ++j) {
                float phi = p / pow(theta, 2.0f * j / d);
                float cos_phi = cos(phi);
                float sin_phi = sin(phi);

                size_t idx_a = i * nhead * d + h * d + j;
                size_t idx_b = i * nhead * d + h * d + j + half_d;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    float a = llaisys::utils::cast<float>(in[idx_a]);
                    float b = llaisys::utils::cast<float>(in[idx_b]);
                    out[idx_a] = llaisys::utils::cast<T>(a * cos_phi - b * sin_phi);
                    out[idx_b] = llaisys::utils::cast<T>(b * cos_phi + a * sin_phi);
                } else {
                    float a = static_cast<float>(in[idx_a]);
                    float b = static_cast<float>(in[idx_b]);
                    out[idx_a] = static_cast<T>(a * cos_phi - b * sin_phi);
                    out[idx_b] = static_cast<T>(b * cos_phi + a * sin_phi);
                }
            }
        }
    }
}

#define DISPATCH_SELF_ATTENTION(dtype, ctype) case dtype: rope_<ctype>(in, out, pos_ids, seqlen, nhead, d, theta); break;

namespace llaisys::ops::cpu {
void rope(std::byte *in, const std::byte *out, const std::byte *pos_ids, llaisysDataType_t type, size_t seqlen, size_t nhead, size_t d, float theta) {
    switch (type) {
        DISPATCH_SELF_ATTENTION(LLAISYS_DTYPE_F32, float)
        DISPATCH_SELF_ATTENTION(LLAISYS_DTYPE_BF16, llaisys::bf16_t)
        DISPATCH_SELF_ATTENTION(LLAISYS_DTYPE_F16, llaisys::fp16_t)
        DISPATCH_SELF_ATTENTION(LLAISYS_DTYPE_I32, int32_t)
        DISPATCH_SELF_ATTENTION(LLAISYS_DTYPE_F64, double)
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
