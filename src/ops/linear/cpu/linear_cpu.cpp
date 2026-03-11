#include "linear_cpu.hpp"
#include "../../../utils.hpp" // 包含类型转换工具
#include <algorithm>
#include <type_traits>
#include <vector>

namespace llaisys::ops::cpu {

// --- 辅助 1：把任意类型转为 float (用于计算) ---
template <typename T>
inline float val_to_float(T v) {
    if constexpr (std::is_same_v<T, float>) {
        return v;
    } else if constexpr (std::is_same_v<T, llaisys::fp16_t>) {
        return llaisys::utils::_f16_to_f32(v);
    } else if constexpr (std::is_same_v<T, llaisys::bf16_t>) {
        return llaisys::utils::_bf16_to_f32(v); // 如果报错，尝试用 _bf16_to_float
    } else {
        return (float)v;
    }
}

// --- 辅助 2：把 float 转回任意类型 (用于存结果) ---
template <typename T>
inline T float_to_val(float v) {
    if constexpr (std::is_same_v<T, float>) {
        return v;
    } else if constexpr (std::is_same_v<T, llaisys::fp16_t>) {
        return llaisys::utils::_f32_to_f16(v);
    } else if constexpr (std::is_same_v<T, llaisys::bf16_t>) {
        return llaisys::utils::_f32_to_bf16(v);
    } else {
        return (T)v;
    }
}

// --- 核心矩阵乘法模板 ---
template <typename T>
void linear_kernel(T *out, const T *in, const T *weight, const T *bias,
                   size_t M, size_t K, size_t N) {
    constexpr size_t BLOCK_N = 32;
    constexpr size_t BLOCK_K = 128;

    // 这里按输出行并行，避免线程之间写同一段 out。
#pragma omp parallel for schedule(static)
    for (ptrdiff_t m = 0; m < static_cast<ptrdiff_t>(M); ++m) {
        for (size_t n0 = 0; n0 < N; n0 += BLOCK_N) {
            size_t n1 = std::min(n0 + BLOCK_N, N);
            float partial[BLOCK_N] = {0.0f};

            for (size_t k0 = 0; k0 < K; k0 += BLOCK_K) {
                size_t k1 = std::min(k0 + BLOCK_K, K);
                for (size_t k = k0; k < k1; ++k) {
                    float x_val = val_to_float(in[m * K + k]);
                    for (size_t n = n0; n < n1; ++n) {
                        partial[n - n0] += x_val * val_to_float(weight[n * K + k]);
                    }
                }
            }

            for (size_t n = n0; n < n1; ++n) {
                float sum = partial[n - n0];
                if (bias) {
                    sum += val_to_float(bias[n]);
                }
                out[m * N + n] = float_to_val<T>(sum);
            }
        }
    }
}

// --- 入口分发函数 ---
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            size_t M, size_t K, size_t N, llaisysDataType_t dtype) {
            
    // 1. Float32
    if (dtype == LLAISYS_DTYPE_F32) {
        linear_kernel<float>(
            (float*)out, (const float*)in, (const float*)weight, (const float*)bias, M, K, N
        );
    }
    // 2. Float16
    else if (dtype == LLAISYS_DTYPE_F16) {
        linear_kernel<llaisys::fp16_t>(
            (llaisys::fp16_t*)out, (const llaisys::fp16_t*)in, (const llaisys::fp16_t*)weight, 
            (const llaisys::fp16_t*)bias, M, K, N
        );
    }
    // 3. BFloat16
    else if (dtype == LLAISYS_DTYPE_BF16) {
        linear_kernel<llaisys::bf16_t>(
            (llaisys::bf16_t*)out, (const llaisys::bf16_t*)in, (const llaisys::bf16_t*)weight, 
            (const llaisys::bf16_t*)bias, M, K, N
        );
    }
}

} // namespace llaisys::ops::cpu
