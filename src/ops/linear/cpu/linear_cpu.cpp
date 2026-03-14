#include "linear_cpu.hpp"

#include "../../../utils.hpp"
#include "../../utils.hpp"

#include <cmath>

#include <cblas.h>

template <typename T>
void linear_(std::byte *out_raw, const std::byte *in_raw, const std::byte *weight_raw, 
            const size_t M, const size_t N, const size_t K, const std::byte *bias_raw,
            llaisysDataType_t dtype) {
    // C[:,M,N] = A[:,M,K] * B[:,K,N]
    // out[:,M,N] = in[:,M,K] * weight[:,N,K]^T + bias[:,M,N]
    T *out = reinterpret_cast<T*>(out_raw);
    const T *in = reinterpret_cast<const T*>(in_raw);
    const T *weight = reinterpret_cast<const T*>(weight_raw);
    const T *bias = reinterpret_cast<const T*>(bias_raw);

    // Fast path when input is already float/double: avoid allocation/casting overhead.
    if constexpr (std::is_same_v<T, float>) {
        float *C = reinterpret_cast<float*>(out);
        const float *A = reinterpret_cast<const float*>(in);
        const float *B = reinterpret_cast<const float*>(weight);

        if (bias != nullptr) {
            // Broadcast bias across all rows.
#ifdef _OPENMP
            #pragma omp parallel for if(M * N > 16384)
#endif
            for (size_t i = 0; i < M; ++i) {
                std::memcpy(C + i * N, bias, N * sizeof(float));
            }
        } else {
            std::memset(C, 0, M * N * sizeof(float));
        }

        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasTrans,
            M, N, K,
            1.0f,
            A, K,
            B, K,
            1.0f,
            C, N
        );
        return;
    }

    if constexpr (std::is_same_v<T, double>) {
        double *C = reinterpret_cast<double*>(out);
        const double *A = reinterpret_cast<const double*>(in);
        const double *B = reinterpret_cast<const double*>(weight);

        if (bias != nullptr) {
#ifdef _OPENMP
            #pragma omp parallel for if(M * N > 16384)
#endif
            for (size_t i = 0; i < M; ++i) {
                std::memcpy(C + i * N, bias, N * sizeof(double));
            }
        } else {
            std::memset(C, 0, M * N * sizeof(double));
        }

        cblas_dgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasTrans,
            M, N, K,
            1.0,
            A, K,
            B, K,
            1.0,
            C, N
        );
        return;
    }

    // OpenBLAS itself only supports float/double. For fp16/bf16/int32 we cast into
    // a float/double buffer for the computation, then cast back to the original type.
    const llaisysDataType_t storage_dtype =
        (dtype == LLAISYS_DTYPE_F64 || dtype == LLAISYS_DTYPE_I32) ? LLAISYS_DTYPE_F64 : LLAISYS_DTYPE_F32;

    llaisys::ops::utils::OpenBlasCapableArray in_aligned(M * K, storage_dtype);
    in_aligned.cast_from(in);
    llaisys::ops::utils::OpenBlasCapableArray weight_aligned(K * N, storage_dtype);
    weight_aligned.cast_from(weight);

    // The output buffer is allocated for OpenBLAS use; we fill it explicitly below.
    llaisys::ops::utils::OpenBlasCapableArray out_aligned(M * N, storage_dtype);

    if (out_aligned.dtype() == LLAISYS_DTYPE_F32) {
        float *A = reinterpret_cast<float*>(in_aligned.data());
        float *B = reinterpret_cast<float*>(weight_aligned.data());
        float *C = reinterpret_cast<float*>(out_aligned.data());

        if (bias != nullptr) {
            out_aligned.broadcast_row(bias, M, N);
        } else {
            out_aligned.zeros();
        }

        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasTrans,
            M, N, K,
            1.0f,
            A, K,
            B, K,
            1.0f,
            C, N
        );
    } else if (out_aligned.dtype() == LLAISYS_DTYPE_F64) {
        double *A = reinterpret_cast<double*>(in_aligned.data());
        double *B = reinterpret_cast<double*>(weight_aligned.data());
        double *C = reinterpret_cast<double*>(out_aligned.data());

        if (bias != nullptr) {
            out_aligned.broadcast_row(bias, M, N);
        } else {
            out_aligned.zeros();
        }

        cblas_dgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasTrans,
            M, N, K,
            1.0,
            A, K,
            B, K,
            1.0,
            C, N
        );
    } else {
        throw std::invalid_argument("Unsupported data type for linear_");
    }

    out_aligned.cast_back(out);
}

#define DISPATCH_LINEAR(dtype, ctype) case dtype: linear_<ctype>(out, in, weight, M, N, K, bias, type); break;

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            const size_t M, const size_t N, const size_t K, llaisysDataType_t type) {
    switch (type) {
        DISPATCH_LINEAR(LLAISYS_DTYPE_F32, float)
        DISPATCH_LINEAR(LLAISYS_DTYPE_BF16, llaisys::bf16_t)
        DISPATCH_LINEAR(LLAISYS_DTYPE_F16, llaisys::fp16_t)
        DISPATCH_LINEAR(LLAISYS_DTYPE_I32, int32_t)
        DISPATCH_LINEAR(LLAISYS_DTYPE_F64, double)
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
