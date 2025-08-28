#include "rms_norm_cpu.hpp"
#include "../../../utils.hpp"

#include <cmath> // sqrtf

namespace {

template <typename T>
inline void rms_norm_impl(T *y,
                          const T *x,
                          const T *w,
                          size_t N, size_t d,
                          float eps) {
    for (size_t i = 0; i < N; ++i) {
        const T *x_row = x + i * d;
        T *y_row = y + i * d;

        // 1) sum of squares in float
        float ss = 0.f;
        for (size_t j = 0; j < d; ++j) {
            float v = llaisys::utils::cast<float>(x_row[j]);
            ss += v * v;
        }
        float mean = ss / (d > 0 ? static_cast<float>(d) : 1.f);
        float inv_rms = 1.0f / std::sqrt(mean + eps);

        // 2) normalize & scale by weight
        for (size_t j = 0; j < d; ++j) {
            float xv = llaisys::utils::cast<float>(x_row[j]);
            float ww = llaisys::utils::cast<float>(w[j]);
            float outv = xv * inv_rms * ww;
            y_row[j] = llaisys::utils::cast<T>(outv);
        }
    }
}

} // anonymous namespace

namespace llaisys::ops::cpu {

void rms_norm(std::byte *out,
              const std::byte *in,
              const std::byte *weight,
              llaisysDataType_t type,
              size_t N, size_t d,
              float eps) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_impl(reinterpret_cast<float *>(out),
                             reinterpret_cast<const float *>(in),
                             reinterpret_cast<const float *>(weight),
                             N, d, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_impl(reinterpret_cast<llaisys::fp16_t *>(out),
                             reinterpret_cast<const llaisys::fp16_t *>(in),
                             reinterpret_cast<const llaisys::fp16_t *>(weight),
                             N, d, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_impl(reinterpret_cast<llaisys::bf16_t *>(out),
                             reinterpret_cast<const llaisys::bf16_t *>(in),
                             reinterpret_cast<const llaisys::bf16_t *>(weight),
                             N, d, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu
