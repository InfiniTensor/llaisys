#include "rope_cpu.hpp"
#include "../../../utils.hpp" 
#include <cmath> 
#include <type_traits>

namespace llaisys::ops::cpu {

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

// --- RoPE 核心计算模板 ---
template<typename T>
void rope_kernel(T *out, const T *in, const int64_t *pos_ids,
                 float theta, size_t seq_len, size_t n_head, size_t head_dim) {
    
    // 每个 (token, head) 是独立任务，适合直接并行展开。
#pragma omp parallel for collapse(2) schedule(static)
    for (ptrdiff_t s = 0; s < static_cast<ptrdiff_t>(seq_len); ++s) {
        for (ptrdiff_t h = 0; h < static_cast<ptrdiff_t>(n_head); ++h) {
            int64_t pos = pos_ids[s];
            size_t offset = s * n_head * head_dim + h * head_dim;
            size_t half_dim = head_dim / 2;
            
            for (size_t j = 0; j < half_dim; ++j) {
                // 【关键修改】使用 double 提高角度计算精度，防止大模型下误差过大
                double freq_exp = (double)(2 * j) / (double)head_dim;
                double freq = 1.0 / std::pow((double)theta, freq_exp);
                double angle = (double)pos * freq;
                
                // 计算 cos/sin 后再转回 float 参与向量运算
                float cos_val = (float)std::cos(angle);
                float sin_val = (float)std::sin(angle);
                
                T val_a_raw = in[offset + j];
                T val_b_raw = in[offset + j + half_dim];
                
                float a = val_to_float(val_a_raw);
                float b = val_to_float(val_b_raw);
                
                // RoPE 旋转公式
                float a_new = a * cos_val - b * sin_val;
                float b_new = b * cos_val + a * sin_val;
                
                out[offset + j] = float_to_val<T>(a_new);
                out[offset + j + half_dim] = float_to_val<T>(b_new);
            }
        }
    }
}

void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          float theta, size_t seq_len, size_t n_head, size_t head_dim,
          llaisysDataType_t dtype) {
    
    const int64_t *pos_ptr = (const int64_t*)pos_ids;

    if (dtype == LLAISYS_DTYPE_F32) {
        rope_kernel<float>((float*)out, (const float*)in, pos_ptr, theta, seq_len, n_head, head_dim);
    } else if (dtype == LLAISYS_DTYPE_F16) {
        rope_kernel<llaisys::fp16_t>((llaisys::fp16_t*)out, (const llaisys::fp16_t*)in, pos_ptr, theta, seq_len, n_head, head_dim);
    } else if (dtype == LLAISYS_DTYPE_BF16) {
        rope_kernel<llaisys::bf16_t>((llaisys::bf16_t*)out, (const llaisys::bf16_t*)in, pos_ptr, theta, seq_len, n_head, head_dim);
    }
}

} // namespace llaisys::ops::cpu
