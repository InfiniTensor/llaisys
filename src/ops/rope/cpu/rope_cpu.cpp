#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

// RoPE (Rotary Position Embedding)
// 对每个向量 x_i = [a_i, b_i] 应用旋转
// φ_{i,j} = p_i / θ^(2j/d)
// a'_{i,j} = a_{i,j}cos(φ_{i,j}) - b_{i,j}sin(φ_{i,j})
// b'_{i,j} = b_{i,j}cos(φ_{i,j}) + a_{i,j}sin(φ_{i,j})
template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids,
           size_t seq_len, size_t n_heads, size_t head_dim, float theta) {
    size_t half_dim = head_dim / 2;

    // 对序列中的每个位置
    for (size_t s = 0; s < seq_len; s++) {
        int64_t pos = pos_ids[s];

        // 对每个注意力头
        for (size_t h = 0; h < n_heads; h++) {
            // 对每个维度对 (a, b)
            for (size_t j = 0; j < half_dim; j++) {
                // 计算角度 φ = pos / θ^(2j/d)
                float freq_exponent = (2.0f * j) / head_dim;
                float freq = pos / std::pow(theta, freq_exponent);
                float cos_freq = std::cos(freq);
                float sin_freq = std::sin(freq);

                // 获取输入的 a 和 b
                size_t idx = s * n_heads * head_dim + h * head_dim;
                size_t a_idx = idx + j;
                size_t b_idx = idx + half_dim + j;

                float a_val, b_val;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    a_val = llaisys::utils::cast<float>(in[a_idx]);
                    b_val = llaisys::utils::cast<float>(in[b_idx]);
                } else {
                    a_val = static_cast<float>(in[a_idx]);
                    b_val = static_cast<float>(in[b_idx]);
                }

                // 应用旋转
                // a' = a*cos(φ) - b*sin(φ)
                // b' = b*cos(φ) + a*sin(φ)
                float a_prime = a_val * cos_freq - b_val * sin_freq;
                float b_prime = b_val * cos_freq + a_val * sin_freq;

                // 存储结果
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    out[a_idx] = llaisys::utils::cast<T>(a_prime);
                    out[b_idx] = llaisys::utils::cast<T>(b_prime);
                } else {
                    out[a_idx] = static_cast<T>(a_prime);
                    out[b_idx] = static_cast<T>(b_prime);
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          llaisysDataType_t type, size_t seq_len, size_t n_heads, size_t head_dim, float theta) {
    const int64_t *pos_ptr = reinterpret_cast<const int64_t *>(pos_ids);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                     pos_ptr, seq_len, n_heads, head_dim, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                     pos_ptr, seq_len, n_heads, head_dim, theta);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                     pos_ptr, seq_len, n_heads, head_dim, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
