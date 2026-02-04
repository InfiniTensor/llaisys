#include "linear_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void linear_(std::byte *out_raw, const std::byte *in_raw, const std::byte *weight_raw, 
            const size_t M, const size_t N, const size_t K, const std::byte *bias_raw) {
    // C[:,M,N] = A[:,M,K] * B[:,K,N]
    // out[:,M,N] = in[:,M,K] * weight[:,N,K]^T + bias[:,M,N]
    T *out = reinterpret_cast<T*>(out_raw);
    const T *in = reinterpret_cast<const T*>(in_raw);
    const T *weight = reinterpret_cast<const T*>(weight_raw);
    const T *bias = reinterpret_cast<const T*>(bias_raw);

    // Initialize out to 0
    for (size_t i = 0; i < M * N; i++) {
        out[i] =  llaisys::utils::cast<T>(0);
    }

    // Only support 2D linear

    /* 多次转换, 精度不够
    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            for (size_t k = 0; k < K; k++) {
                // out[m][n] += in[m][k] * weight[n][k]
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    out[m * N + n] = llaisys::utils::cast<T>(
                            llaisys::utils::cast<float>(out[m * N + n])
                            + llaisys::utils::cast<float>(in[m *K + k]) 
                            * llaisys::utils::cast<float>(weight[n * K + k])
                        );
                } else {
                    out[m * N + n] += in[m *K + k] * weight[n * K + k];
                }
            }
        }
    }

    bias 只有一维, 要广播
    if (bias != nullptr) {
        for (size_t i = 0; i < M * N; i++) {
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                out[i] = llaisys::utils::cast<T>(
                            llaisys::utils::cast<float>(out[i])
                            + llaisys::utils::cast<float>(bias[i])
                        );
            } else {
                out[i] += bias[i];
            }
        }
    }

    if (bias != nullptr) {
        for (size_t m = 0; m < M; m++) {
            for (size_t n = 0; n < N; n++) {
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    out[m * N + n] = llaisys::utils::cast<T>(
                                llaisys::utils::cast<float>(out[m * N + n])
                                + llaisys::utils::cast<float>(bias[n])
                            );
                } else {
                    out[m * N + n] += bias[n];
                }
            }
        }
    }
    */

    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; k++) {
                sum += llaisys::utils::cast<float>(in[m * K + k]) * llaisys::utils::cast<float>(weight[n * K + k]);
            }
            if (bias != nullptr) {
                sum += llaisys::utils::cast<float>(bias[n]);
            }
            out[m * N + n] = llaisys::utils::cast<T>(sum);
        }
    }
}

#define DISPATCH_LINEAR(dtype, ctype) case dtype: linear_<ctype>(out, in, weight, M, N, K, bias); break;

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
