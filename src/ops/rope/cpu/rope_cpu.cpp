#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cassert>

// RoPE (Rotary Position Embedding) 实现
// 输入形状: [seq_len, n_heads, head_dim]
// pos_ids 形状: [seq_len] (int64)
template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids,
           size_t seq_len, size_t n_heads, size_t head_dim, float theta) {
    
    // head_dim 必须是偶数 (这在上层已经检查过了)
    // assert(head_dim % 2 == 0);
    
    size_t half_dim = head_dim / 2;
    
    for (size_t s = 0; s < seq_len; s++) {
        // 获取当前位置ID
        float position = static_cast<float>(pos_ids[s]);
        
        for (size_t h = 0; h < n_heads; h++) {
            // 计算当前头的输入和输出偏移
            const T *head_in = in + (s * n_heads + h) * head_dim;
            T *head_out = out + (s * n_heads + h) * head_dim;
            
            // 处理每一对(a, b)
            for (size_t i = 0; i < half_dim; i++) {
                // 计算旋转频率，使用与PyTorch相同的方式: freqs = positions / (theta ** (2 * i / head_dim))
                float freq_exp = (2.0f * static_cast<float>(i)) / static_cast<float>(head_dim);
                float freq_base = std::pow(theta, freq_exp);
                float angle = position / freq_base;
                
                float cos_val = std::cos(angle);
                float sin_val = std::sin(angle);
                
                // 获取输入的a和b值
                float a_val, b_val;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    a_val = llaisys::utils::cast<float>(head_in[i]);
                    b_val = llaisys::utils::cast<float>(head_in[i + half_dim]);
                } else {
                    a_val = static_cast<float>(head_in[i]);
                    b_val = static_cast<float>(head_in[i + half_dim]);
                }
                
                // 应用旋转：
                // a' = a * cos - b * sin
                // b' = b * cos + a * sin
                float a_new = a_val * cos_val - b_val * sin_val;
                float b_new = b_val * cos_val + a_val * sin_val;
                
                // 存储结果
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    head_out[i] = llaisys::utils::cast<T>(a_new);
                    head_out[i + half_dim] = llaisys::utils::cast<T>(b_new);
                } else {
                    head_out[i] = static_cast<T>(a_new);
                    head_out[i + half_dim] = static_cast<T>(b_new);
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, 
          llaisysDataType_t type, size_t seq_len, size_t n_heads, size_t head_dim, float theta) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const int64_t *>(pos_ids),
            seq_len, n_heads, head_dim, theta
        );
    case LLAISYS_DTYPE_BF16:
        return rope_(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            reinterpret_cast<const int64_t *>(pos_ids),
            seq_len, n_heads, head_dim, theta
        );
    case LLAISYS_DTYPE_F16:
        return rope_(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            reinterpret_cast<const int64_t *>(pos_ids),
            seq_len, n_heads, head_dim, theta
        );
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
