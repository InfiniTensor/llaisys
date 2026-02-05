#include "rope.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <vector>

template <typename T>
static void rope_(
    T* out, const T* in, const int64_t* pos_id, float theta,
    size_t seqlen, size_t nhead, size_t d
) {
    using namespace llaisys::utils;
    size_t half = d / 2;

    std::vector<double> inv_freq(half);
    for (size_t j = 0; j < half; j++) {
        inv_freq[j] = 1.0 / std::pow(theta, 2.0 * j / d);
    }

    for (size_t i = 0; i < seqlen; i++) {
        double p = static_cast<double>(pos_id[i]);
        for (size_t h = 0; h < nhead; h++) {
            size_t base = (i * nhead + h) * d;
            for (size_t j = 0; j < half; j++) {
                double angle = p * inv_freq[j];
                double cos_val = std::cos(angle);
                double sin_val = std::sin(angle);

                double a = static_cast<double>(cast<float>(in[base + j]));
                double b = static_cast<double>(cast<float>(in[base + j + half]));

                out[base + j]        = cast<T>(a * cos_val - b * sin_val);
                out[base + j + half] = cast<T>(b * cos_val + a * sin_val);
            }
        }
    }
}

namespace llaisys::ops::cpu {
    void rope(
        void* out,
        const void* in,
        const void* pos_id,
        float theta,
        size_t seqlen,
        size_t nhead,
        size_t d,
        llaisysDataType_t dtype
    ) {
        switch (dtype) {
            case LLAISYS_DTYPE_F32:
                return rope_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                             reinterpret_cast<const int64_t *>(pos_id), theta, seqlen, nhead, d);
            case LLAISYS_DTYPE_BF16:
                return rope_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                             reinterpret_cast<const int64_t *>(pos_id), theta, seqlen, nhead, d);
            case LLAISYS_DTYPE_F16:
                return rope_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                             reinterpret_cast<const int64_t *>(pos_id), theta, seqlen, nhead, d);
            default:
                EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
        }
    }
} // namespace llaisys::ops::cpu