#include "swiglu_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <type_traits>

namespace llaisys::ops::cpu {

// --- 类型转换辅助 (Standard) ---
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

// --- SwiGLU 核心计算模板 ---
template<typename T>
void swiglu_kernel(T *out, const T *gate, const T *up, size_t numel) {
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(numel); ++i) {
        float g_val = val_to_float(gate[i]);
        float u_val = val_to_float(up[i]);

        // 计算 SiLU(g) = g / (1 + e^-g)
        float silu_g = g_val / (1.0f + std::exp(-g_val));

        // out = up * SiLU(g)
        float res = u_val * silu_g;

        out[i] = float_to_val<T>(res);
    }
}

// --- 入口分发 ---
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, 
            size_t numel, llaisysDataType_t dtype) {
    
    if (dtype == LLAISYS_DTYPE_F32) {
        swiglu_kernel<float>((float*)out, (const float*)gate, (const float*)up, numel);
    } else if (dtype == LLAISYS_DTYPE_F16) {
        swiglu_kernel<llaisys::fp16_t>((llaisys::fp16_t*)out, (const llaisys::fp16_t*)gate, (const llaisys::fp16_t*)up, numel);
    } else if (dtype == LLAISYS_DTYPE_BF16) {
        swiglu_kernel<llaisys::bf16_t>((llaisys::bf16_t*)out, (const llaisys::bf16_t*)gate, (const llaisys::bf16_t*)up, numel);
    }
}

} // namespace llaisys::ops::cpu
