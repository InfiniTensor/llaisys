#include "rope_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids,
           size_t seq_len, size_t n_heads, size_t head_dim, float theta) {
    // RoPE: Rotary Position Embedding
    // Split input into two halves: [a0, a1, ..., a_{d/2-1}, b0, b1, ..., b_{d/2-1}]
    // For each position and each dimension pair:
    // a'[j] = a[j] * cos(angle) - b[j] * sin(angle)
    // b'[j] = b[j] * cos(angle) + a[j] * sin(angle)
    // where angle = pos / (theta^(2j/d))

    size_t half_dim = head_dim / 2;

    for (size_t s = 0; s < seq_len; s++) {
        float pos = static_cast<float>(pos_ids[s]);

        for (size_t h = 0; h < n_heads; h++) {
            for (size_t j = 0; j < half_dim; j++) {
                // Calculate angle: pos / (theta^(2j/d)) to match PyTorch exactly
                // Using the same calculation order as PyTorch
                float exponent = (2.0f * static_cast<float>(j)) / static_cast<float>(head_dim);
                float divisor = std::pow(theta, exponent);
                float angle = pos / divisor;
                float cos_angle = std::cos(angle);
                float sin_angle = std::sin(angle);

                // Get indices - first half is 'a', second half is 'b'
                size_t idx_a = s * n_heads * head_dim + h * head_dim + j;
                size_t idx_b = s * n_heads * head_dim + h * head_dim + half_dim + j;

                float a, b;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    a = llaisys::utils::cast<float>(in[idx_a]);
                    b = llaisys::utils::cast<float>(in[idx_b]);
                } else {
                    a = static_cast<float>(in[idx_a]);
                    b = static_cast<float>(in[idx_b]);
                }

                // Apply rotation
                float a_new = a * cos_angle - b * sin_angle;
                float b_new = b * cos_angle + a * sin_angle;

                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    out[idx_a] = llaisys::utils::cast<T>(a_new);
                    out[idx_b] = llaisys::utils::cast<T>(b_new);
                } else {
                    out[idx_a] = static_cast<T>(a_new);
                    out[idx_b] = static_cast<T>(b_new);
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          llaisysDataType_t dtype, size_t seq_len, size_t n_heads, size_t head_dim, float theta) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rope_<float>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const int64_t *>(pos_ids),
            seq_len, n_heads, head_dim, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_<llaisys::bf16_t>(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            reinterpret_cast<const int64_t *>(pos_ids),
            seq_len, n_heads, head_dim, theta);
    case LLAISYS_DTYPE_F16:
        return rope_<llaisys::fp16_t>(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            reinterpret_cast<const int64_t *>(pos_ids),
            seq_len, n_heads, head_dim, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
