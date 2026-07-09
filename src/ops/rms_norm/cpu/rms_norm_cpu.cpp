#include "rms_norm_cpu.hpp"
#include "../../../utils.hpp" // 包含类型转换工具
#include <cmath> // sqrt
#include <type_traits>

namespace llaisys::ops::cpu {

// --- 类型转换辅助函数 (复用 Linear 的逻辑) ---
template <typename T>
inline float val_to_float(T v) {
    if constexpr (std::is_same_v<T, float>) {
        return v;
    } else if constexpr (std::is_same_v<T, llaisys::fp16_t>) {
        return llaisys::utils::_f16_to_f32(v);
    } else if constexpr (std::is_same_v<T, llaisys::bf16_t>) {
        return llaisys::utils::_bf16_to_f32(v);
    } else {
        return (float)v;
    }
}

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

// --- RMSNorm 核心计算模板 ---
template<typename T>
void rms_norm_kernel(T *out, const T *in, const T *weight,
                     float eps, size_t num_rows, size_t hidden_size) {
    
    // 遍历每一行 (每一个 token)
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(num_rows); ++i) {
        const T *row_in = in + i * hidden_size;
        T *row_out = out + i * hidden_size;

        // 1. 计算平方和 (Sum of Squares)
        float sum_sq = 0.0f;
        for (size_t j = 0; j < hidden_size; ++j) {
            float val = val_to_float(row_in[j]);
            sum_sq += val * val;
        }

        // 2. 计算 RMS 的倒数 (Inverse RMS)
        // mean = sum_sq / hidden_size
        // rms = sqrt(mean + eps)
        // inv_rms = 1 / rms
        float mean_sq = sum_sq / hidden_size;
        float inv_rms = 1.0f / std::sqrt(mean_sq + eps);

        // 3. 归一化并应用权重 (Normalize & Scale)
        for (size_t j = 0; j < hidden_size; ++j) {
            float val = val_to_float(row_in[j]);
            float w = val_to_float(weight[j]);
            
            // 公式: out = (in * inv_rms) * weight
            float res = val * inv_rms * w;
            
            row_out[j] = float_to_val<T>(res);
        }
    }
}

// --- 入口分发函数 ---
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              float eps, size_t num_rows, size_t hidden_size,
              llaisysDataType_t dtype) {
    
    if (dtype == LLAISYS_DTYPE_F32) {
        rms_norm_kernel<float>(
            (float*)out, (const float*)in, (const float*)weight,
            eps, num_rows, hidden_size
        );
    } else if (dtype == LLAISYS_DTYPE_F16) {
        rms_norm_kernel<llaisys::fp16_t>(
            (llaisys::fp16_t*)out, (const llaisys::fp16_t*)in, (const llaisys::fp16_t*)weight,
            eps, num_rows, hidden_size
        );
    } else if (dtype == LLAISYS_DTYPE_BF16) {
        rms_norm_kernel<llaisys::bf16_t>(
            (llaisys::bf16_t*)out, (const llaisys::bf16_t*)in, (const llaisys::bf16_t*)weight,
            eps, num_rows, hidden_size
        );
    }
}

} // namespace llaisys::ops::cpu
