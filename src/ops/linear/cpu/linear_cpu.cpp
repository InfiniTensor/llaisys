#include "linear_cpu.hpp"

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

#ifdef LLAISYS_USE_OPENBLAS
extern "C" {
#include <cblas.h>
}
#endif

#include "../../../utils.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace {

constexpr std::ptrdiff_t OUTPUT_TILE = 32;
constexpr std::ptrdiff_t OUTPUT_UNROLL = 4;
constexpr std::ptrdiff_t ROW_UNROLL = 2;
constexpr std::ptrdiff_t REDUCTION_TILE = 256;

#ifdef LLAISYS_USE_OPENBLAS
inline bool should_use_openblas(const std::vector<size_t> &shapes) {
    const std::ptrdiff_t dimi = static_cast<std::ptrdiff_t>(shapes[0]);
    const std::ptrdiff_t dimk = static_cast<std::ptrdiff_t>(shapes[1]);
    const std::ptrdiff_t dimj = static_cast<std::ptrdiff_t>(shapes[2]);
    return dimi >= 16 && dimk >= 1024 && dimj >= 1024 && dimk >= dimj;
}
#endif

inline bool should_use_gemm_kernel(const std::vector<size_t> &shapes) {
    const std::ptrdiff_t dimi = static_cast<std::ptrdiff_t>(shapes[0]);
    const std::ptrdiff_t dimk = static_cast<std::ptrdiff_t>(shapes[1]);
    const std::ptrdiff_t dimj = static_cast<std::ptrdiff_t>(shapes[2]);
    return dimi >= 16 && dimk >= 1024 && dimj >= 1024;
}

#if (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)) && (defined(__GNUC__) || defined(__clang__))
inline bool has_avx2_fma() {
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}

__attribute__((target("avx2,fma,sse3")))
inline float hsum256_ps(__m256 v) {
    const __m128 low = _mm256_castps256_ps128(v);
    const __m128 high = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

__attribute__((target("avx2,fma")))
void linear_f32_avx2_matvec(float *out, const float *in, const float *weight, const float *bias, const std::vector<size_t> &shapes) {
    const std::ptrdiff_t dimi = static_cast<std::ptrdiff_t>(shapes[0]);
    const std::ptrdiff_t dimk = static_cast<std::ptrdiff_t>(shapes[1]);
    const std::ptrdiff_t dimj = static_cast<std::ptrdiff_t>(shapes[2]);
    const std::ptrdiff_t ksimd = dimk - (dimk % 8);

#pragma omp parallel for collapse(2) schedule(static)
    for (std::ptrdiff_t i = 0; i < dimi; ++i) {
        for (std::ptrdiff_t j0 = 0; j0 < dimj; j0 += OUTPUT_TILE) {
            const float *xrow = in + i * dimk;
            float *outrow = out + i * dimj;
            const std::ptrdiff_t jend = std::min(j0 + OUTPUT_TILE, dimj);
            std::ptrdiff_t j = j0;

            for (; j + OUTPUT_UNROLL <= jend; j += OUTPUT_UNROLL) {
                const float *w0 = weight + (j + 0) * dimk;
                const float *w1 = weight + (j + 1) * dimk;
                const float *w2 = weight + (j + 2) * dimk;
                const float *w3 = weight + (j + 3) * dimk;

                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                __m256 acc2 = _mm256_setzero_ps();
                __m256 acc3 = _mm256_setzero_ps();

                for (std::ptrdiff_t k = 0; k < ksimd; k += 8) {
                    const __m256 x = _mm256_loadu_ps(xrow + k);
                    acc0 = _mm256_fmadd_ps(x, _mm256_loadu_ps(w0 + k), acc0);
                    acc1 = _mm256_fmadd_ps(x, _mm256_loadu_ps(w1 + k), acc1);
                    acc2 = _mm256_fmadd_ps(x, _mm256_loadu_ps(w2 + k), acc2);
                    acc3 = _mm256_fmadd_ps(x, _mm256_loadu_ps(w3 + k), acc3);
                }

                float sum0 = hsum256_ps(acc0) + (bias ? bias[j + 0] : 0.0f);
                float sum1 = hsum256_ps(acc1) + (bias ? bias[j + 1] : 0.0f);
                float sum2 = hsum256_ps(acc2) + (bias ? bias[j + 2] : 0.0f);
                float sum3 = hsum256_ps(acc3) + (bias ? bias[j + 3] : 0.0f);

                for (std::ptrdiff_t k = ksimd; k < dimk; ++k) {
                    const float x = xrow[k];
                    sum0 += x * w0[k];
                    sum1 += x * w1[k];
                    sum2 += x * w2[k];
                    sum3 += x * w3[k];
                }

                outrow[j + 0] = sum0;
                outrow[j + 1] = sum1;
                outrow[j + 2] = sum2;
                outrow[j + 3] = sum3;
            }

            for (; j < jend; ++j) {
                const float *wrow = weight + j * dimk;
                __m256 acc = _mm256_setzero_ps();

                for (std::ptrdiff_t k = 0; k < ksimd; k += 8) {
                    const __m256 x = _mm256_loadu_ps(xrow + k);
                    acc = _mm256_fmadd_ps(x, _mm256_loadu_ps(wrow + k), acc);
                }

                float sum = hsum256_ps(acc) + (bias ? bias[j] : 0.0f);
                for (std::ptrdiff_t k = ksimd; k < dimk; ++k) {
                    sum += xrow[k] * wrow[k];
                }
                outrow[j] = sum;
            }
        }
    }
}

__attribute__((target("avx2,fma")))
void linear_f32_avx2_gemm(float *out, const float *in, const float *weight, const float *bias, const std::vector<size_t> &shapes) {
    const std::ptrdiff_t dimi = static_cast<std::ptrdiff_t>(shapes[0]);
    const std::ptrdiff_t dimk = static_cast<std::ptrdiff_t>(shapes[1]);
    const std::ptrdiff_t dimj = static_cast<std::ptrdiff_t>(shapes[2]);
    const std::ptrdiff_t ksimd = dimk - (dimk % 8);

#pragma omp parallel for collapse(2) schedule(static)
    for (std::ptrdiff_t i0 = 0; i0 < dimi; i0 += ROW_UNROLL) {
        for (std::ptrdiff_t j0 = 0; j0 < dimj; j0 += OUTPUT_TILE) {
            const std::ptrdiff_t iend = std::min(i0 + ROW_UNROLL, dimi);
            const std::ptrdiff_t jend = std::min(j0 + OUTPUT_TILE, dimj);

            if (iend - i0 < ROW_UNROLL) {
                for (std::ptrdiff_t i = i0; i < iend; ++i) {
                    const float *xrow = in + i * dimk;
                    float *outrow = out + i * dimj;
                    std::ptrdiff_t j = j0;
                    for (; j + OUTPUT_UNROLL <= jend; j += OUTPUT_UNROLL) {
                        const float *w0 = weight + (j + 0) * dimk;
                        const float *w1 = weight + (j + 1) * dimk;
                        const float *w2 = weight + (j + 2) * dimk;
                        const float *w3 = weight + (j + 3) * dimk;

                        __m256 acc0 = _mm256_setzero_ps();
                        __m256 acc1 = _mm256_setzero_ps();
                        __m256 acc2 = _mm256_setzero_ps();
                        __m256 acc3 = _mm256_setzero_ps();

                        for (std::ptrdiff_t k = 0; k < ksimd; k += 8) {
                            const __m256 x = _mm256_loadu_ps(xrow + k);
                            acc0 = _mm256_fmadd_ps(x, _mm256_loadu_ps(w0 + k), acc0);
                            acc1 = _mm256_fmadd_ps(x, _mm256_loadu_ps(w1 + k), acc1);
                            acc2 = _mm256_fmadd_ps(x, _mm256_loadu_ps(w2 + k), acc2);
                            acc3 = _mm256_fmadd_ps(x, _mm256_loadu_ps(w3 + k), acc3);
                        }

                        float sum0 = hsum256_ps(acc0) + (bias ? bias[j + 0] : 0.0f);
                        float sum1 = hsum256_ps(acc1) + (bias ? bias[j + 1] : 0.0f);
                        float sum2 = hsum256_ps(acc2) + (bias ? bias[j + 2] : 0.0f);
                        float sum3 = hsum256_ps(acc3) + (bias ? bias[j + 3] : 0.0f);
                        for (std::ptrdiff_t k = ksimd; k < dimk; ++k) {
                            const float x = xrow[k];
                            sum0 += x * w0[k];
                            sum1 += x * w1[k];
                            sum2 += x * w2[k];
                            sum3 += x * w3[k];
                        }

                        outrow[j + 0] = sum0;
                        outrow[j + 1] = sum1;
                        outrow[j + 2] = sum2;
                        outrow[j + 3] = sum3;
                    }

                    for (; j < jend; ++j) {
                        const float *wrow = weight + j * dimk;
                        __m256 acc = _mm256_setzero_ps();
                        for (std::ptrdiff_t k = 0; k < ksimd; k += 8) {
                            acc = _mm256_fmadd_ps(_mm256_loadu_ps(xrow + k), _mm256_loadu_ps(wrow + k), acc);
                        }
                        float sum = hsum256_ps(acc) + (bias ? bias[j] : 0.0f);
                        for (std::ptrdiff_t k = ksimd; k < dimk; ++k) {
                            sum += xrow[k] * wrow[k];
                        }
                        outrow[j] = sum;
                    }
                }
                continue;
            }

            const float *xrow0 = in + i0 * dimk;
            const float *xrow1 = in + (i0 + 1) * dimk;
            float *outrow0 = out + i0 * dimj;
            float *outrow1 = out + (i0 + 1) * dimj;
            std::ptrdiff_t j = j0;

            for (; j + OUTPUT_UNROLL <= jend; j += OUTPUT_UNROLL) {
                const float *w0 = weight + (j + 0) * dimk;
                const float *w1 = weight + (j + 1) * dimk;
                const float *w2 = weight + (j + 2) * dimk;
                const float *w3 = weight + (j + 3) * dimk;

                __m256 acc00 = _mm256_setzero_ps();
                __m256 acc01 = _mm256_setzero_ps();
                __m256 acc02 = _mm256_setzero_ps();
                __m256 acc03 = _mm256_setzero_ps();
                __m256 acc10 = _mm256_setzero_ps();
                __m256 acc11 = _mm256_setzero_ps();
                __m256 acc12 = _mm256_setzero_ps();
                __m256 acc13 = _mm256_setzero_ps();

                for (std::ptrdiff_t k = 0; k < ksimd; k += 8) {
                    const __m256 x0 = _mm256_loadu_ps(xrow0 + k);
                    const __m256 x1 = _mm256_loadu_ps(xrow1 + k);
                    const __m256 wv0 = _mm256_loadu_ps(w0 + k);
                    const __m256 wv1 = _mm256_loadu_ps(w1 + k);
                    const __m256 wv2 = _mm256_loadu_ps(w2 + k);
                    const __m256 wv3 = _mm256_loadu_ps(w3 + k);

                    acc00 = _mm256_fmadd_ps(x0, wv0, acc00);
                    acc01 = _mm256_fmadd_ps(x0, wv1, acc01);
                    acc02 = _mm256_fmadd_ps(x0, wv2, acc02);
                    acc03 = _mm256_fmadd_ps(x0, wv3, acc03);
                    acc10 = _mm256_fmadd_ps(x1, wv0, acc10);
                    acc11 = _mm256_fmadd_ps(x1, wv1, acc11);
                    acc12 = _mm256_fmadd_ps(x1, wv2, acc12);
                    acc13 = _mm256_fmadd_ps(x1, wv3, acc13);
                }

                float sum00 = hsum256_ps(acc00) + (bias ? bias[j + 0] : 0.0f);
                float sum01 = hsum256_ps(acc01) + (bias ? bias[j + 1] : 0.0f);
                float sum02 = hsum256_ps(acc02) + (bias ? bias[j + 2] : 0.0f);
                float sum03 = hsum256_ps(acc03) + (bias ? bias[j + 3] : 0.0f);
                float sum10 = hsum256_ps(acc10) + (bias ? bias[j + 0] : 0.0f);
                float sum11 = hsum256_ps(acc11) + (bias ? bias[j + 1] : 0.0f);
                float sum12 = hsum256_ps(acc12) + (bias ? bias[j + 2] : 0.0f);
                float sum13 = hsum256_ps(acc13) + (bias ? bias[j + 3] : 0.0f);

                for (std::ptrdiff_t k = ksimd; k < dimk; ++k) {
                    const float x0 = xrow0[k];
                    const float x1 = xrow1[k];
                    const float wv0 = w0[k];
                    const float wv1 = w1[k];
                    const float wv2 = w2[k];
                    const float wv3 = w3[k];
                    sum00 += x0 * wv0;
                    sum01 += x0 * wv1;
                    sum02 += x0 * wv2;
                    sum03 += x0 * wv3;
                    sum10 += x1 * wv0;
                    sum11 += x1 * wv1;
                    sum12 += x1 * wv2;
                    sum13 += x1 * wv3;
                }

                outrow0[j + 0] = sum00;
                outrow0[j + 1] = sum01;
                outrow0[j + 2] = sum02;
                outrow0[j + 3] = sum03;
                outrow1[j + 0] = sum10;
                outrow1[j + 1] = sum11;
                outrow1[j + 2] = sum12;
                outrow1[j + 3] = sum13;
            }

            for (; j < jend; ++j) {
                const float *wrow = weight + j * dimk;
                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                for (std::ptrdiff_t k = 0; k < ksimd; k += 8) {
                    const __m256 wv = _mm256_loadu_ps(wrow + k);
                    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(xrow0 + k), wv, acc0);
                    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(xrow1 + k), wv, acc1);
                }
                float sum0 = hsum256_ps(acc0) + (bias ? bias[j] : 0.0f);
                float sum1 = hsum256_ps(acc1) + (bias ? bias[j] : 0.0f);
                for (std::ptrdiff_t k = ksimd; k < dimk; ++k) {
                    const float wv = wrow[k];
                    sum0 += xrow0[k] * wv;
                    sum1 += xrow1[k] * wv;
                }
                outrow0[j] = sum0;
                outrow1[j] = sum1;
            }
        }
    }
}
#endif

#ifdef LLAISYS_USE_OPENBLAS
void linear_f32_openblas(float *out, const float *in, const float *weight, const float *bias, const std::vector<size_t> &shapes) {
    const blasint dimi = static_cast<blasint>(shapes[0]);
    const blasint dimk = static_cast<blasint>(shapes[1]);
    const blasint dimj = static_cast<blasint>(shapes[2]);

    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasTrans,
        dimi,
        dimj,
        dimk,
        1.0f,
        in,
        dimk,
        weight,
        dimk,
        0.0f,
        out,
        dimj);

    if (bias == nullptr) {
        return;
    }

    const std::ptrdiff_t dimi_p = static_cast<std::ptrdiff_t>(dimi);
    const std::ptrdiff_t dimj_p = static_cast<std::ptrdiff_t>(dimj);
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < dimi_p; ++i) {
        float *outrow = out + i * dimj_p;
        for (std::ptrdiff_t j0 = 0; j0 < dimj_p; j0 += OUTPUT_TILE) {
            const std::ptrdiff_t jend = std::min(j0 + OUTPUT_TILE, dimj_p);
#pragma omp simd
            for (std::ptrdiff_t j = j0; j < jend; ++j) {
                outrow[j] += bias[j];
            }
        }
    }
}
#endif

void linear_f32(float *out, const float *in, const float *weight, const float *bias, const std::vector<size_t> &shapes) {
    const std::ptrdiff_t dimi = static_cast<std::ptrdiff_t>(shapes[0]);
    const std::ptrdiff_t dimk = static_cast<std::ptrdiff_t>(shapes[1]);
    const std::ptrdiff_t dimj = static_cast<std::ptrdiff_t>(shapes[2]);

#pragma omp parallel for collapse(2) schedule(static)
    for (std::ptrdiff_t i = 0; i < dimi; ++i) {
        for (std::ptrdiff_t j0 = 0; j0 < dimj; j0 += OUTPUT_TILE) {
            const float *xrow = in + i * dimk;
            float *outrow = out + i * dimj;
            const std::ptrdiff_t jend = std::min(j0 + OUTPUT_TILE, dimj);
            std::ptrdiff_t j = j0;

            for (; j + OUTPUT_UNROLL <= jend; j += OUTPUT_UNROLL) {
                const float *w0 = weight + (j + 0) * dimk;
                const float *w1 = weight + (j + 1) * dimk;
                const float *w2 = weight + (j + 2) * dimk;
                const float *w3 = weight + (j + 3) * dimk;

                float acc0 = bias ? bias[j + 0] : 0.0f;
                float acc1 = bias ? bias[j + 1] : 0.0f;
                float acc2 = bias ? bias[j + 2] : 0.0f;
                float acc3 = bias ? bias[j + 3] : 0.0f;

                for (std::ptrdiff_t k0 = 0; k0 < dimk; k0 += REDUCTION_TILE) {
                    const std::ptrdiff_t kend = std::min(k0 + REDUCTION_TILE, dimk);
                    float part0 = 0.0f;
                    float part1 = 0.0f;
                    float part2 = 0.0f;
                    float part3 = 0.0f;

#pragma omp simd reduction(+ : part0, part1, part2, part3)
                    for (std::ptrdiff_t k = k0; k < kend; ++k) {
                        const float x = xrow[k];
                        part0 += x * w0[k];
                        part1 += x * w1[k];
                        part2 += x * w2[k];
                        part3 += x * w3[k];
                    }

                    acc0 += part0;
                    acc1 += part1;
                    acc2 += part2;
                    acc3 += part3;
                }

                outrow[j + 0] = acc0;
                outrow[j + 1] = acc1;
                outrow[j + 2] = acc2;
                outrow[j + 3] = acc3;
            }

            for (; j < jend; ++j) {
                const float *wrow = weight + j * dimk;
                float acc = bias ? bias[j] : 0.0f;

                for (std::ptrdiff_t k0 = 0; k0 < dimk; k0 += REDUCTION_TILE) {
                    const std::ptrdiff_t kend = std::min(k0 + REDUCTION_TILE, dimk);
                    float part = 0.0f;

#pragma omp simd reduction(+ : part)
                    for (std::ptrdiff_t k = k0; k < kend; ++k) {
                        part += xrow[k] * wrow[k];
                    }

                    acc += part;
                }

                outrow[j] = acc;
            }
        }
    }
}

template <typename T>
void linear_generic(T *out, const T *in, const T *weight, const T *bias, const std::vector<size_t> &shapes) {
    const std::ptrdiff_t dimi = static_cast<std::ptrdiff_t>(shapes[0]);
    const std::ptrdiff_t dimk = static_cast<std::ptrdiff_t>(shapes[1]);
    const std::ptrdiff_t dimj = static_cast<std::ptrdiff_t>(shapes[2]);

#pragma omp parallel for collapse(2) schedule(static)
    for (std::ptrdiff_t i = 0; i < dimi; ++i) {
        for (std::ptrdiff_t j = 0; j < dimj; ++j) {
            float sum = 0.0f;
            for (std::ptrdiff_t k = 0; k < dimk; ++k) {
                sum += llaisys::utils::cast<float>(in[i * dimk + k]) * llaisys::utils::cast<float>(weight[j * dimk + k]);
            }
            if (bias != nullptr) {
                sum += llaisys::utils::cast<float>(bias[j]);
            }
            out[i * dimj + j] = llaisys::utils::cast<T>(sum);
        }
    }
}

} // namespace

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, llaisysDataType_t type, std::vector<size_t> shapes){
    switch (type) {
    case LLAISYS_DTYPE_F32:
#ifdef LLAISYS_USE_OPENBLAS
        if (should_use_openblas(shapes)) {
            return linear_f32_openblas(
                reinterpret_cast<float *>(out),
                reinterpret_cast<const float *>(in),
                reinterpret_cast<const float *>(weight),
                bias ? reinterpret_cast<const float *>(bias) : nullptr,
                shapes);
        }
#endif
#if (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)) && (defined(__GNUC__) || defined(__clang__))
        if (has_avx2_fma()) {
            if (should_use_gemm_kernel(shapes)) {
                return linear_f32_avx2_gemm(
                    reinterpret_cast<float *>(out),
                    reinterpret_cast<const float *>(in),
                    reinterpret_cast<const float *>(weight),
                    bias ? reinterpret_cast<const float *>(bias) : nullptr,
                    shapes);
            }
            return linear_f32_avx2_matvec(
                reinterpret_cast<float *>(out),
                reinterpret_cast<const float *>(in),
                reinterpret_cast<const float *>(weight),
                bias ? reinterpret_cast<const float *>(bias) : nullptr,
                shapes);
        }
#endif
        return linear_f32(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight),
            bias ? reinterpret_cast<const float *>(bias) : nullptr,
            shapes);
    case LLAISYS_DTYPE_BF16:
        return linear_generic(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            reinterpret_cast<const llaisys::bf16_t *>(weight),
            bias ? reinterpret_cast<const llaisys::bf16_t *>(bias) : nullptr,
            shapes);
    case LLAISYS_DTYPE_F16:
        return linear_generic(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            reinterpret_cast<const llaisys::fp16_t *>(weight),
            bias ? reinterpret_cast<const llaisys::fp16_t *>(bias) : nullptr,
            shapes);
    default:    
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
