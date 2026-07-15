#include "rms_norm_cpu.hpp"

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#ifdef __C
#pragma push_macro("__C")
#undef __C
#define LLAISYS_RESTORE_C_MACRO
#endif
#include <immintrin.h>
#ifdef LLAISYS_RESTORE_C_MACRO
#pragma pop_macro("__C")
#undef LLAISYS_RESTORE_C_MACRO
#endif
#endif

#include "../../../utils.hpp"

#include <cstddef>
#include <cmath>

namespace {

#if (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)) && (defined(__GNUC__) || defined(__clang__))
inline bool has_avx2() {
    return __builtin_cpu_supports("avx2");
}

__attribute__((target("avx2,sse3")))
inline float hsum256_ps(__m256 v) {
    const __m128 low = _mm256_castps256_ps128(v);
    const __m128 high = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

__attribute__((target("avx2")))
void rms_norm_f32_avx2(float *out, const float *in, const float *weight, float eps, const std::vector<size_t> &shapes) {
    const std::ptrdiff_t dimi = static_cast<std::ptrdiff_t>(shapes[0]);
    const std::ptrdiff_t dimj = static_cast<std::ptrdiff_t>(shapes[1]);
    const std::ptrdiff_t simd_dimj = dimj - (dimj % 8);
    const bool parallel_rows = dimi > 1 && dimi * dimj >= 65536;

#pragma omp parallel for schedule(static) if (parallel_rows)
    for (std::ptrdiff_t i = 0; i < dimi; ++i) {
        const float *in_row = in + i * dimj;
        float *out_row = out + i * dimj;
        __m256 sum_acc = _mm256_setzero_ps();

        for (std::ptrdiff_t j = 0; j < simd_dimj; j += 8) {
            const __m256 x = _mm256_loadu_ps(in_row + j);
            sum_acc = _mm256_add_ps(sum_acc, _mm256_mul_ps(x, x));
        }

        float sum_sq = hsum256_ps(sum_acc);
        for (std::ptrdiff_t j = simd_dimj; j < dimj; ++j) {
            sum_sq += in_row[j] * in_row[j];
        }

        const float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(dimj) + eps);
        const __m256 inv = _mm256_set1_ps(inv_rms);

        std::ptrdiff_t j = 0;
        for (; j < simd_dimj; j += 8) {
            const __m256 x = _mm256_loadu_ps(in_row + j);
            const __m256 w = _mm256_loadu_ps(weight + j);
            _mm256_storeu_ps(out_row + j, _mm256_mul_ps(_mm256_mul_ps(x, inv), w));
        }

        for (; j < dimj; ++j) {
            out_row[j] = in_row[j] * inv_rms * weight[j];
        }
    }
}
#endif

void rms_norm_f32(float *out, const float *in, const float *weight, float eps, const std::vector<size_t> &shapes) {
    const std::ptrdiff_t dimi = static_cast<std::ptrdiff_t>(shapes[0]);
    const std::ptrdiff_t dimj = static_cast<std::ptrdiff_t>(shapes[1]);
    const bool parallel_rows = dimi > 1 && dimi * dimj >= 65536;

#pragma omp parallel for schedule(static) if (parallel_rows)
    for (std::ptrdiff_t i = 0; i < dimi; ++i) {
        const float *in_row = in + i * dimj;
        float *out_row = out + i * dimj;
        float sum_sq = 0.0f;

#pragma omp simd reduction(+ : sum_sq)
        for (std::ptrdiff_t j = 0; j < dimj; ++j) {
            sum_sq += in_row[j] * in_row[j];
        }

        const float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(dimj) + eps);

#pragma omp simd
        for (std::ptrdiff_t j = 0; j < dimj; ++j) {
            out_row[j] = in_row[j] * inv_rms * weight[j];
        }
    }
}

template <typename T>
void rms_norm_generic(T *out, const T *in, const T *weight, float eps, const std::vector<size_t> &shapes) {
    const std::ptrdiff_t dimi = static_cast<std::ptrdiff_t>(shapes[0]);
    const std::ptrdiff_t dimj = static_cast<std::ptrdiff_t>(shapes[1]);
    const bool parallel_rows = dimi > 1 && dimi * dimj >= 65536;

#pragma omp parallel for schedule(static) if (parallel_rows)
    for (std::ptrdiff_t i = 0; i < dimi; ++i) {
        const T *in_row = in + i * dimj;
        T *out_row = out + i * dimj;
        float sum_sq = 0.0f;
        for (std::ptrdiff_t j = 0; j < dimj; ++j) {
            const float val = llaisys::utils::cast<float>(in_row[j]);
            sum_sq += val * val;
        }

        const float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(dimj) + eps);
        for (std::ptrdiff_t j = 0; j < dimj; ++j) {
            const float val = llaisys::utils::cast<float>(in_row[j]);
            const float w = llaisys::utils::cast<float>(weight[j]);
            out_row[j] = llaisys::utils::cast<T>(val * inv_rms * w);
        }
    }
}

} // namespace

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, const float eps, llaisysDataType_t type, std::vector<size_t> shapes){
    switch (type) {
    case LLAISYS_DTYPE_F32:
#if (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)) && (defined(__GNUC__) || defined(__clang__))
        if (has_avx2()) {
            return rms_norm_f32_avx2(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(weight), eps, shapes);
        }
#endif
        return rms_norm_f32(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(weight), eps, shapes);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_generic(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in), reinterpret_cast<const llaisys::bf16_t *>(weight), eps, shapes);
    case LLAISYS_DTYPE_F16:
        return rms_norm_generic(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in), reinterpret_cast<const llaisys::fp16_t *>(weight), eps, shapes);
    default:    
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
