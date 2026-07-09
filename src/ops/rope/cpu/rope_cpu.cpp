#include "rope_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, size_t seqlen, size_t nhead, size_t d, float theta) {
    size_t half_d = d / 2;
    for (size_t s = 0; s < seqlen; s++) {
        float pos = static_cast<float>(pos_ids[s]);
        for (size_t h = 0; h < nhead; h++) {
            for (size_t j = 0; j < half_d; j++) {
                float angle = pos / std::pow(theta, 2.0f * j / d);
                float cos_val = std::cos(angle);
                float sin_val = std::sin(angle);
                
                size_t idx = s * nhead * d + h * d;
                float a = llaisys::utils::cast<float>(in[idx + j]);
                float b = llaisys::utils::cast<float>(in[idx + j + half_d]);
                
                out[idx + j] = llaisys::utils::cast<T>(a * cos_val - b * sin_val);
                out[idx + j + half_d] = llaisys::utils::cast<T>(b * cos_val + a * sin_val);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, llaisysDataType_t type,
          size_t seqlen, size_t nhead, size_t d, float theta) {
    auto ids = reinterpret_cast<const int64_t *>(pos_ids);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), ids, seqlen, nhead, d, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in), ids, seqlen, nhead, d, theta);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in), ids, seqlen, nhead, d, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
}
