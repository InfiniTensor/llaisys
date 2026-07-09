#ifdef __AVX2__
#include <immintrin.h>
#endif

#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, float eps, size_t rows, size_t cols) {
    #pragma omp parallel for schedule(dynamic)
    for (size_t r = 0; r < rows; r++) {
        const T *row_in = in + r * cols;
        T *row_out = out + r * cols;

        float sum_sq = 0.0f;

#ifdef __AVX2__
        if constexpr (std::is_same_v<T, float>) {
            __m256 vsum = _mm256_setzero_ps();
            size_t c = 0;
            for (; c + 8 <= cols; c += 8) {
                __m256 vx = _mm256_loadu_ps(row_in + c);
                vsum = _mm256_fmadd_ps(vx, vx, vsum);
            }
            float tmp[8];
            _mm256_storeu_ps(tmp, vsum);
            sum_sq = tmp[0] + tmp[1] + tmp[2] + tmp[3] +
                     tmp[4] + tmp[5] + tmp[6] + tmp[7];
            for (; c < cols; c++) {
                float v = row_in[c];
                sum_sq += v * v;
            }
        } else {
            for (size_t c = 0; c < cols; c++) {
                float v = llaisys::utils::cast<float>(row_in[c]);
                sum_sq += v * v;
            }
        }
#else
        for (size_t c = 0; c < cols; c++) {
            float v = llaisys::utils::cast<float>(row_in[c]);
            sum_sq += v * v;
        }
#endif

        float rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(cols) + eps);

#ifdef __AVX2__
        if constexpr (std::is_same_v<T, float>) {
            __m256 vrms = _mm256_set1_ps(rms);
            size_t c = 0;
            for (; c + 8 <= cols; c += 8) {
                __m256 vx = _mm256_loadu_ps(row_in + c);
                __m256 vw = _mm256_loadu_ps(reinterpret_cast<const float *>(weight) + c);
                __m256 vout = _mm256_mul_ps(_mm256_mul_ps(vw, vx), vrms);
                _mm256_storeu_ps(row_out + c, vout);
            }
            for (; c < cols; c++) {
                row_out[c] = weight[c] * row_in[c] * rms;
            }
        } else {
            for (size_t c = 0; c < cols; c++) {
                float v = llaisys::utils::cast<float>(row_in[c]);
                float w = llaisys::utils::cast<float>(weight[c]);
                row_out[c] = llaisys::utils::cast<T>(w * v * rms);
            }
        }
#else
        for (size_t c = 0; c < cols; c++) {
            float v = llaisys::utils::cast<float>(row_in[c]);
            float w = llaisys::utils::cast<float>(weight[c]);
            row_out[c] = llaisys::utils::cast<T>(w * v * rms);
        }
#endif
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              float eps, llaisysDataType_t dtype, size_t rows, size_t cols) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out),
                          reinterpret_cast<const float *>(in),
                          reinterpret_cast<const float *>(weight), eps, rows, cols);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out),
                          reinterpret_cast<const llaisys::bf16_t *>(in),
                          reinterpret_cast<const llaisys::bf16_t *>(weight), eps, rows, cols);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out),
                          reinterpret_cast<const llaisys::fp16_t *>(in),
                          reinterpret_cast<const llaisys::fp16_t *>(weight), eps, rows, cols);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
