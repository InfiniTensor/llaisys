#include "argmax_cpu.hpp"
#include "../../../utils.hpp" // 包含 types.hpp
#include <vector>
#include <limits>
#include <algorithm>
#include <type_traits> // 用于 std::is_same_v

namespace llaisys::ops::cpu {

// --- 辅助：数值转换 ---
template <typename T>
inline float val_to_float(T v) {
    if constexpr (std::is_same_v<T, float>) {
        return v;
    } 
    else if constexpr (std::is_same_v<T, llaisys::fp16_t>) {
        // 这里的函数名来自 types.cpp 的实现
        return llaisys::utils::_f16_to_f32(v);
    }
    else if constexpr (std::is_same_v<T, llaisys::bf16_t>) {
        // 这里的函数名是根据命名惯例推导的，如果报错改为 _bf16_to_f32
        return llaisys::utils::_bf16_to_f32(v);
    }
    else {
        return (float)v; // Fallback
    }
}

// 核心计算模板
template<typename T, typename IDX_T>
void argmax_kernel(IDX_T *out_idx, T *out_val, const T *vals, size_t numel) {
    // 初始化最大值 (用 float 负无穷)
    float max_v = -std::numeric_limits<float>::infinity();
    size_t max_i = 0;

    for (size_t i = 0; i < numel; ++i) {
        float val = val_to_float(vals[i]);
        if (val > max_v) {
            max_v = val;
            max_i = i;
        }
    }

    // 将结果存回
    out_idx[0] = (IDX_T)max_i;
    out_val[0] = vals[max_i]; 
}

// 入口函数
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
            size_t numel, 
            llaisysDataType_t dtype, 
            llaisysDataType_t idx_dtype) {
            
    // 1. Float32
    if (dtype == LLAISYS_DTYPE_F32) {
        if (idx_dtype == LLAISYS_DTYPE_I64) {
            argmax_kernel<float, int64_t>((int64_t*)max_idx, (float*)max_val, (const float*)vals, numel);
        } else {
            argmax_kernel<float, int32_t>((int32_t*)max_idx, (float*)max_val, (const float*)vals, numel);
        }
    }
    // 2. Float16
    else if (dtype == LLAISYS_DTYPE_F16) {
        if (idx_dtype == LLAISYS_DTYPE_I64) {
            argmax_kernel<llaisys::fp16_t, int64_t>(
                (int64_t*)max_idx, (llaisys::fp16_t*)max_val, (const llaisys::fp16_t*)vals, numel
            );
        } else {
             argmax_kernel<llaisys::fp16_t, int32_t>(
                (int32_t*)max_idx, (llaisys::fp16_t*)max_val, (const llaisys::fp16_t*)vals, numel
            );
        }
    }
    // 3. BFloat16
    else if (dtype == LLAISYS_DTYPE_BF16) {
        if (idx_dtype == LLAISYS_DTYPE_I64) {
            argmax_kernel<llaisys::bf16_t, int64_t>(
                (int64_t*)max_idx, (llaisys::bf16_t*)max_val, (const llaisys::bf16_t*)vals, numel
            );
        } else {
             argmax_kernel<llaisys::bf16_t, int32_t>(
                (int32_t*)max_idx, (llaisys::bf16_t*)max_val, (const llaisys::bf16_t*)vals, numel
            );
        }
    }
}

} // namespace llaisys::ops::cpu