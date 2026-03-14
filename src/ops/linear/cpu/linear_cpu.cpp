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

    llaisys::ops::utils::OpenBlasCapableArray in_aligned(M*K, in, dtype);
    llaisys::ops::utils::OpenBlasCapableArray weight_aligned(K*N, weight, dtype);
    
    // out_aligned should initially read out to have correct output size, but we really want output as out array
    // Wait, since OpenBlasCapableArray performs heap allocation and reads input memory:
    // It's cheaper to initialize it with out_raw (even if uninitialized) instead of OOB bias read
    llaisys::ops::utils::OpenBlasCapableArray out_aligned(M*N, out, dtype);

    if (out_aligned.dtype() == LLAISYS_DTYPE_F32) {
        float *A = reinterpret_cast<float*>(in_aligned.data());
        float *B = reinterpret_cast<float*>(weight_aligned.data());
        float *C = reinterpret_cast<float*>(out_aligned.data());

        if (bias != nullptr) {
            llaisys::ops::utils::OpenBlasCapableArray bias_aligned(N, bias, dtype);
            float *b_ptr = reinterpret_cast<float*>(bias_aligned.data());
            for (size_t i = 0; i < M; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    C[i * N + j] = b_ptr[j];
                }
            }
        } else {
            for (size_t i = 0; i < M * N; ++i) C[i] = 0.0f;
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
            llaisys::ops::utils::OpenBlasCapableArray bias_aligned(N, bias, dtype);
            double *b_ptr = reinterpret_cast<double*>(bias_aligned.data());
            for (size_t i = 0; i < M; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    C[i * N + j] = b_ptr[j];
                }
            }
        } else {
            for (size_t i = 0; i < M * N; ++i) C[i] = 0.0;
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
