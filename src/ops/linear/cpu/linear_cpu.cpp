#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef USE_OPENBLAS
#include <cblas.h>
#endif

#include "linear_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstring>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_OPENBLAS

static void linear_f32_blas(float *out, const float *in, const float *weight,
                            const float *bias, size_t M, size_t N, size_t K, bool has_bias) {
    // out[M,N] = in[M,K] * weight[N,K]^T + bias[N]
    if (has_bias) {
        for (size_t m = 0; m < M; m++) {
            std::memcpy(out + m * N, bias, N * sizeof(float));
        }
        scipy_cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                           (blasint)M, (blasint)N, (blasint)K,
                           1.0f, in, (blasint)K, weight, (blasint)K,
                           1.0f, out, (blasint)N);
    } else {
        scipy_cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                           (blasint)M, (blasint)N, (blasint)K,
                           1.0f, in, (blasint)K, weight, (blasint)K,
                           0.0f, out, (blasint)N);
    }
}

#else // !USE_OPENBLAS

#ifdef __AVX2__

static void linear_f32_avx2(float *out, const float *in, const float *weight,
                             const float *bias, size_t M, size_t N, size_t K, bool has_bias) {
    #pragma omp parallel for schedule(dynamic)
    for (size_t m = 0; m < M; m++) {
        const float *a_row = in + m * K;
        float *c_row = out + m * N;

        for (size_t n = 0; n < N; n++) {
            const float *b_row = weight + n * K;
            __m256 vsum = _mm256_setzero_ps();
            size_t k = 0;

            for (; k + 8 <= K; k += 8) {
                __m256 va = _mm256_loadu_ps(a_row + k);
                __m256 vb = _mm256_loadu_ps(b_row + k);
                vsum = _mm256_fmadd_ps(va, vb, vsum);
            }

            float tmp[8];
            _mm256_storeu_ps(tmp, vsum);
            float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] +
                        tmp[4] + tmp[5] + tmp[6] + tmp[7];

            for (; k < K; k++) {
                sum += a_row[k] * b_row[k];
            }

            if (has_bias) sum += bias[n];
            c_row[n] = sum;
        }
    }
}

#endif // __AVX2__

#endif // USE_OPENBLAS

template <typename T>
void linear_generic(T *out, const T *in, const T *weight, const T *bias,
                    size_t M, size_t N, size_t K, bool has_bias) {
    // Convert to F32, compute, convert back
    std::vector<float> f_in(M * K), f_weight(N * K), f_out(M * N);
    std::vector<float> f_bias;
    if (has_bias) f_bias.resize(N);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < M * K; i++)
        f_in[i] = llaisys::utils::cast<float>(in[i]);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N * K; i++)
        f_weight[i] = llaisys::utils::cast<float>(weight[i]);

    if (has_bias) {
        for (size_t i = 0; i < N; i++)
            f_bias[i] = llaisys::utils::cast<float>(bias[i]);
    }

#ifdef USE_OPENBLAS
    linear_f32_blas(f_out.data(), f_in.data(), f_weight.data(),
                    has_bias ? f_bias.data() : nullptr, M, N, K, has_bias);
#elif defined(__AVX2__)
    linear_f32_avx2(f_out.data(), f_in.data(), f_weight.data(),
                    has_bias ? f_bias.data() : nullptr, M, N, K, has_bias);
#else
    // Fallback naive
    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; k++)
                sum += f_in[m * K + k] * f_weight[n * K + k];
            if (has_bias) sum += f_bias[n];
            f_out[m * N + n] = sum;
        }
    }
#endif

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < M * N; i++)
        out[i] = llaisys::utils::cast<T>(f_out[i]);
}

static void linear_f32(float *out, const float *in, const float *weight,
                       const float *bias, size_t M, size_t N, size_t K, bool has_bias) {
#ifdef USE_OPENBLAS
    linear_f32_blas(out, in, weight, bias, M, N, K, has_bias);
#elif defined(__AVX2__)
    linear_f32_avx2(out, in, weight, bias, M, N, K, has_bias);
#else
    // Fallback: naive with OpenMP
    #pragma omp parallel for schedule(dynamic)
    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; k++)
                sum += in[m * K + k] * weight[n * K + k];
            if (has_bias) sum += bias[n];
            out[m * N + n] = sum;
        }
    }
#endif
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t dtype, size_t M, size_t N, size_t K, bool has_bias) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return linear_f32(reinterpret_cast<float *>(out),
                          reinterpret_cast<const float *>(in),
                          reinterpret_cast<const float *>(weight),
                          has_bias ? reinterpret_cast<const float *>(bias) : nullptr,
                          M, N, K, has_bias);
    case LLAISYS_DTYPE_BF16:
        return linear_generic(reinterpret_cast<llaisys::bf16_t *>(out),
                              reinterpret_cast<const llaisys::bf16_t *>(in),
                              reinterpret_cast<const llaisys::bf16_t *>(weight),
                              has_bias ? reinterpret_cast<const llaisys::bf16_t *>(bias) : nullptr,
                              M, N, K, has_bias);
    case LLAISYS_DTYPE_F16:
        return linear_generic(reinterpret_cast<llaisys::fp16_t *>(out),
                              reinterpret_cast<const llaisys::fp16_t *>(in),
                              reinterpret_cast<const llaisys::fp16_t *>(weight),
                              has_bias ? reinterpret_cast<const llaisys::fp16_t *>(bias) : nullptr,
                              M, N, K, has_bias);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
