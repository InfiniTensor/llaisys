#include "../../../utils.hpp"
#include <cmath>
#include <cstddef>
#include <type_traits>
#include "rope_cpu.hpp"
template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, float theta, size_t seq_len, size_t n_heads, size_t head_dim) {
    size_t half_dim = head_dim / 2;
    for (size_t i = 0; i < seq_len; ++i) {
        auto pos = pos_ids[i];
        for (size_t j = 0; j < n_heads; ++j) {
            size_t offset = i * head_dim * n_heads + j * head_dim;
            for (size_t k = 0; k < half_dim; ++k) {
                float angle = pos * 1.0f / std::pow(theta, 2.0f*k/head_dim);
                float sin_angle = std::sin(angle);
                float cos_angle = std::cos(angle);

                auto a_i_j = in[offset + k];
                auto b_i_j = in[offset + k + half_dim];
                if constexpr (std::is_same_v<T, float>) {
                    out[offset + k] = a_i_j * cos_angle - b_i_j * sin_angle;
                    out[offset + k + half_dim] = b_i_j * cos_angle + a_i_j * sin_angle;
                } else {
                    auto a = llaisys::utils::cast<float>(a_i_j);
                    auto b = llaisys::utils::cast<float>(b_i_j);
                    out[offset + k] = llaisys::utils::cast<T>(a * cos_angle - b * sin_angle);
                    out[offset + k + half_dim] = llaisys::utils::cast<T>(b * cos_angle + a * sin_angle);
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, float theta, llaisysDataType_t dtype, size_t seq_len, size_t n_heads, size_t head_dim) {
    switch (dtype) {
        case LLAISYS_DTYPE_F32:
            return rope_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const int64_t *>(pos_ids), theta, seq_len, n_heads, head_dim);
        case LLAISYS_DTYPE_BF16:
            return rope_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in), reinterpret_cast<const int64_t *>(pos_ids), theta, seq_len, n_heads, head_dim);
        case LLAISYS_DTYPE_F16:
            return rope_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in), reinterpret_cast<const int64_t *>(pos_ids), theta, seq_len, n_heads, head_dim);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu