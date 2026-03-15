#include "linear_cpu.hpp"

#include "../../../utils.hpp"
#include "llaisys.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#ifdef LLAISYS_USE_OPENBLAS
#if __has_include(<openblas/cblas.h>)
#include <openblas/cblas.h>
#define LLAISYS_HAS_CBLAS 1
#elif __has_include(<cblas.h>)
#include <cblas.h>
#define LLAISYS_HAS_CBLAS 1
#endif
#endif

// 分块矩阵乘 (F32)，提升 cache 命中，无 OpenBLAS 时使用
static constexpr size_t kBlock = 64u;

static void linear_f32_blocked(float *out,
                               const float *in,
                               const float *weight,
                               const float *bias,
                               size_t M,
                               size_t N,
                               size_t K) {
    if (bias != nullptr) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (M >= 64)
#endif
        for (int i = 0; i < static_cast<int>(M); i++) {
            for (size_t j = 0; j < N; j++) {
                out[i * N + j] = bias[j];
            }
        }
    } else {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (M >= 64)
#endif
        for (int i = 0; i < static_cast<int>(M); i++) {
            for (size_t j = 0; j < N; j++) {
                out[i * N + j] = 0.0f;
            }
        }
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (M >= 2 * kBlock)
#endif
    for (int ib = 0; ib < static_cast<int>(M); ib += static_cast<int>(kBlock)) {
        size_t ie = (std::min)(ib + kBlock, M);
        for (size_t kb = 0; kb < K; kb += kBlock) {
            size_t ke = (std::min)(kb + kBlock, K);
            for (size_t jb = 0; jb < N; jb += kBlock) {
                size_t je = (std::min)(jb + kBlock, N);
                for (size_t i = ib; i < ie; i++) {
                    for (size_t j = jb; j < je; j++) {
                        float sum = out[i * N + j];
                        for (size_t k = kb; k < ke; k++) {
                            sum += in[i * K + k] * weight[j * K + k];
                        }
                        out[i * N + j] = sum;
                    }
                }
            }
        }
    }
}

// 通用内核：按外积方式实现 Y = X W^T + b（BF16/F16 或无 OpenBLAS 时使用）
template <typename T>
static void linear_naive(T *out,
                         const T *in,
                         const T *weight,
                         const T *bias,
                         size_t M,
                         size_t N,
                         size_t K) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (M >= 64)
#endif
    for (int i = 0; i < static_cast<int>(M); i++) {
        for (size_t j = 0; j < N; j++) {
            float sum = 0.0f;
            if (bias != nullptr) {
                sum += llaisys::utils::cast<float>(bias[j]);
            }
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                for (size_t k = 0; k < K; k++) {
                    float data_x = llaisys::utils::cast<float>(in[i * K + k]);
                    float data_w = llaisys::utils::cast<float>(weight[j * K + k]);
                    sum += data_x * data_w;
                }
                out[i * N + j] = llaisys::utils::cast<T>(sum);
            } else {
                for (size_t k = 0; k < K; k++) {
                    sum += in[i * K + k] * weight[j * K + k];
                }
                out[i * N + j] = sum;
            }
        }
    }
}

#if defined(LLAISYS_USE_OPENBLAS) && defined(LLAISYS_HAS_CBLAS)
// F32: 直接调用 SGEMM，再加 bias
static void linear_f32_openblas(float *out,
                                const float *in,
                                const float *weight,
                                const float *bias,
                                size_t M,
                                size_t N,
                                size_t K) {
    // C = alpha * A * B^T + beta * C   =>  out = 1 * in * weight^T + 0 * out
    // RowMajor: A[M,K] lda=K, B[N,K] transB => B^T[K,N] ldb=K, C[M,N] ldc=N
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                (int)M, (int)N, (int)K,
                1.0f, in, (int)K, weight, (int)K, 0.0f, out, (int)N);
    if (bias != nullptr) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (M >= 64)
#endif
        for (int i = 0; i < static_cast<int>(M); i++) {
            for (size_t j = 0; j < N; j++) {
                out[i * N + j] += bias[j];
            }
        }
    }
}

// BF16/F16: 分块转 float -> SGEMM -> 转回，避免整块临时矩阵过大
static constexpr size_t kLinearBlockRows = 256;

template <typename T>
static void linear_bf16_f16_openblas(T *out,
                                    const T *in,
                                    const T *weight,
                                    const T *bias,
                                    size_t M,
                                    size_t N,
                                    size_t K) {
    std::vector<float> w_float(static_cast<size_t>(N) * K);
    for (size_t j = 0; j < N; j++) {
        for (size_t k = 0; k < K; k++) {
            w_float[j * K + k] = llaisys::utils::cast<float>(weight[j * K + k]);
        }
    }
    std::vector<float> in_block(kLinearBlockRows * K);
    std::vector<float> out_block(kLinearBlockRows * N);

    for (size_t i0 = 0; i0 < M; i0 += kLinearBlockRows) {
        size_t rows = (std::min)(i0 + kLinearBlockRows, M) - i0;
        for (size_t i = 0; i < rows; i++) {
            for (size_t k = 0; k < K; k++) {
                in_block[i * K + k] = llaisys::utils::cast<float>(in[(i0 + i) * K + k]);
            }
        }
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    (int)rows, (int)N, (int)K,
                    1.0f, in_block.data(), (int)K, w_float.data(), (int)K,
                    0.0f, out_block.data(), (int)N);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < N; j++) {
                float v = out_block[i * N + j];
                if (bias != nullptr) {
                    v += llaisys::utils::cast<float>(bias[j]);
                }
                out[(i0 + i) * N + j] = llaisys::utils::cast<T>(v);
            }
        }
    }
}
#endif // LLAISYS_USE_OPENBLAS && LLAISYS_HAS_CBLAS

namespace llaisys::ops::cpu {
void linear(std::byte *out,
            const std::byte *in,
            const std::byte *weight,
            const std::byte *bias,
            llaisysDataType_t type,
            size_t M,
            size_t N,
            size_t K) {
#if defined(LLAISYS_USE_OPENBLAS) && defined(LLAISYS_HAS_CBLAS)
    if (type == LLAISYS_DTYPE_F32) {
        return linear_f32_openblas(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight),
            reinterpret_cast<const float *>(bias),
            M, N, K);
    }
    if (type == LLAISYS_DTYPE_BF16) {
        return linear_bf16_f16_openblas<bf16_t>(
            reinterpret_cast<bf16_t *>(out),
            reinterpret_cast<const bf16_t *>(in),
            reinterpret_cast<const bf16_t *>(weight),
            reinterpret_cast<const bf16_t *>(bias),
            M, N, K);
    }
    if (type == LLAISYS_DTYPE_F16) {
        return linear_bf16_f16_openblas<fp16_t>(
            reinterpret_cast<fp16_t *>(out),
            reinterpret_cast<const fp16_t *>(in),
            reinterpret_cast<const fp16_t *>(weight),
            reinterpret_cast<const fp16_t *>(bias),
            M, N, K);
    }
#else
    (void)M;
    (void)N;
    (void)K;
#endif
    switch (type) {
    case LLAISYS_DTYPE_F16:
        return linear_naive(reinterpret_cast<fp16_t *>(out),
                           reinterpret_cast<const fp16_t *>(in),
                           reinterpret_cast<const fp16_t *>(weight),
                           reinterpret_cast<const fp16_t *>(bias),
                           M, N, K);
    case LLAISYS_DTYPE_BF16:
        return linear_naive(reinterpret_cast<bf16_t *>(out),
                           reinterpret_cast<const bf16_t *>(in),
                           reinterpret_cast<const bf16_t *>(weight),
                           reinterpret_cast<const bf16_t *>(bias),
                           M, N, K);
    case LLAISYS_DTYPE_F32:
        return linear_f32_blocked(reinterpret_cast<float *>(out),
                                  reinterpret_cast<const float *>(in),
                                  reinterpret_cast<const float *>(weight),
                                  reinterpret_cast<const float *>(bias),
                                  M, N, K);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
