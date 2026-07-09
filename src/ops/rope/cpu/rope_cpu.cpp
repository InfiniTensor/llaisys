#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids,
           float theta, size_t seqlen, size_t nhead, size_t d) {
    size_t half_d = d / 2;

    // Precompute theta powers to avoid redundant pow() calls per element
    std::vector<float> theta_pow(half_d);
    for (size_t j = 0; j < half_d; j++) {
        theta_pow[j] = std::pow(theta, 2.0f * static_cast<float>(j) / static_cast<float>(d));
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t s = 0; s < seqlen; s++) {
        for (size_t h = 0; h < nhead; h++) {
            float pos = static_cast<float>(pos_ids[s]);
            const T *x = in + (s * nhead + h) * d;
            T *y = out + (s * nhead + h) * d;
            const T *a = x;
            const T *b = x + half_d;
            T *a_out = y;
            T *b_out = y + half_d;

            for (size_t j = 0; j < half_d; j++) {
                float phi = pos / theta_pow[j];
                float cos_phi = std::cos(phi);
                float sin_phi = std::sin(phi);
                float a_val = llaisys::utils::cast<float>(a[j]);
                float b_val = llaisys::utils::cast<float>(b[j]);
                a_out[j] = llaisys::utils::cast<T>(a_val * cos_phi - b_val * sin_phi);
                b_out[j] = llaisys::utils::cast<T>(b_val * cos_phi + a_val * sin_phi);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          float theta, llaisysDataType_t dtype,
          size_t seqlen, size_t nhead, size_t d) {
    auto *pids = reinterpret_cast<const int64_t *>(pos_ids);
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                      pids, theta, seqlen, nhead, d);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                      pids, theta, seqlen, nhead, d);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                      pids, theta, seqlen, nhead, d);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
