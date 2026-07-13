#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstdint>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, float theta, size_t seq_len, size_t n_heads, size_t head_dim) {
    const size_t half_dim = head_dim / 2;
    // Precompute inverse frequencies (denominators) for each dimension j
    // To match PyTorch's behavior: phi = pos / (theta ^ (2j/d))
    std::vector<float> inv_freqs(half_dim);
    for (size_t j = 0; j < half_dim; ++j) {
        inv_freqs[j] = static_cast<float>(std::pow(static_cast<double>(theta), 2.0 * static_cast<double>(j) / static_cast<double>(head_dim)));
    }

    for (size_t i = 0; i < seq_len; ++i) {
        const float pos = static_cast<float>(pos_ids[i]);
        for (size_t h = 0; h < n_heads; ++h) {
            const size_t base_idx = (i * n_heads + h) * head_dim;
            const T* in_ptr = in + base_idx;
            T* out_ptr = out + base_idx;
            for (size_t j = 0; j < half_dim; ++j) {
                float phi = pos / inv_freqs[j];
                float cos_phi = std::cos(phi);
                float sin_phi = std::sin(phi);
                // a = in[j], b = in[j + half_dim]
                float a = llaisys::utils::cast<float>(in_ptr[j]);
                float b = llaisys::utils::cast<float>(in_ptr[j + half_dim]);
                float a_prime = a * cos_phi - b * sin_phi;
                float b_prime = b * cos_phi + a * sin_phi;
                out_ptr[j] = llaisys::utils::cast<T>(a_prime);
                out_ptr[j + half_dim] = llaisys::utils::cast<T>(b_prime);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const int64_t *pos_ids, float theta, llaisysDataType_t type, size_t seq_len, size_t n_heads, size_t head_dim) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), pos_ids, theta, seq_len, n_heads, head_dim);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in), pos_ids, theta, seq_len, n_heads, head_dim);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in), pos_ids, theta, seq_len, n_heads, head_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
