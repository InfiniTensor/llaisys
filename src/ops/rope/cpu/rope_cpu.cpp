#include "rope_cpu.hpp"
#include "../../../utils.hpp"

#include <cmath> // std::powf, std::cosf, std::sinf
#include <vector>

namespace {

template <typename T>
inline void rope_impl(T *out,
                      const T *in,
                      const int64_t *pos,
                      size_t seqlen, size_t nhead, size_t d,
                      float theta_f) {
    const size_t half = d / 2;
    const double theta = static_cast<double>(theta_f);

    // inv_freq[j] = theta^{-(2j/d)}
    std::vector<double> inv_freq(half);
    const double denom = static_cast<double>(half);// d/2
    for (size_t j = 0; j < half; ++j) {
        const double exp = -static_cast<double>(j) / denom;
        inv_freq[j] = std::pow(theta, exp);
        //  1.0f / std::pow(theta, 2.0f*j/d)
    }

    // layout: contiguous row-major
    // vector offset step per (i,h): d elements
    for (size_t i = 0; i < seqlen; ++i) {
        const double p = static_cast<double>(pos[i]);
        for (size_t h = 0; h < nhead; ++h) {
            const T *src = in + (i * nhead + h) * d;
            T *dst = out + (i * nhead + h) * d;

            for (size_t j = 0; j < half; ++j) {
                const double a = static_cast<double>(llaisys::utils::cast<float>(src[j]));
                const double b = static_cast<double>(llaisys::utils::cast<float>(src[j + half]));

                const double angle = p * inv_freq[j];
                const double c = std::cos(angle);
                const double s = std::sin(angle);

                const double ap = a * c - b * s;
                const double bp = b * c + a * s;

                dst[j] = llaisys::utils::cast<T>(static_cast<float>(ap));
                dst[j + half] = llaisys::utils::cast<T>(static_cast<float>(bp));
            }
        }
    }
}

} // anonymous namespace

namespace llaisys::ops::cpu {

void rope(std::byte *out,
          const std::byte *in,
          const int64_t *pos,
          llaisysDataType_t type,
          size_t seqlen, size_t nhead, size_t d,
          float theta) {
    const size_t half = d / 2;
    ASSERT(d % 2 == 0 && half > 0, "rope: last dimension d must be even and > 0");

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_impl(reinterpret_cast<float *>(out),
                         reinterpret_cast<const float *>(in),
                         pos, seqlen, nhead, d, theta);
    case LLAISYS_DTYPE_F16:
        return rope_impl(reinterpret_cast<llaisys::fp16_t *>(out),
                         reinterpret_cast<const llaisys::fp16_t *>(in),
                         pos, seqlen, nhead, d, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_impl(reinterpret_cast<llaisys::bf16_t *>(out),
                         reinterpret_cast<const llaisys::bf16_t *>(in),
                         pos, seqlen, nhead, d, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu
