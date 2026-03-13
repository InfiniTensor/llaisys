#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstring>
#include <cstdint>

namespace llaisys::ops::cpu {

template <typename T>
void rope_(std::byte *out_bytes, const std::byte *in_bytes, const std::byte *pos_ids_bytes, float theta, size_t seq_len, size_t num_heads, size_t head_dim) {
    auto out = reinterpret_cast<T *>(out_bytes);
    auto in = reinterpret_cast<const T *>(in_bytes);
    auto pos_ids = reinterpret_cast<const int64_t *>(pos_ids_bytes);

    size_t half_dim = head_dim / 2;

    for (size_t i = 0; i < seq_len; ++i) {
        int64_t p = pos_ids[i];
        
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t j = 0; j < half_dim; ++j) {
                // Compute rotation angle following PyTorch implementation
                // freqs = positions / (theta ** (2 * i / head_dim))
                float dim = static_cast<float>(j);
                float exponent = 2.0f * dim / head_dim;
                float theta_pow = std::pow(theta, exponent);
                float phi = static_cast<float>(p) / theta_pow;
                
                // Compute sin and cos
                float cos_phi = std::cos(phi);
                float sin_phi = std::sin(phi);
                
                // Get input values [a, b]
                size_t index_a = i * num_heads * head_dim + h * head_dim + j;
                size_t index_b = index_a + half_dim;
                
                T a = in[index_a];
                T b = in[index_b];
                
                // Apply rotation
                float a_f = llaisys::utils::cast<float>(a);
                float b_f = llaisys::utils::cast<float>(b);
                
                float a_prime = a_f * cos_phi - b_f * sin_phi;
                float b_prime = b_f * cos_phi + a_f * sin_phi;
                
                // Store result
                out[index_a] = llaisys::utils::cast<T>(a_prime);
                out[index_b] = llaisys::utils::cast<T>(b_prime);
            }
        }
    }
}

void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, llaisysDataType_t type, float theta, size_t seq_len, size_t num_heads, size_t head_dim) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_<float>(out, in, pos_ids, theta, seq_len, num_heads, head_dim);
    case LLAISYS_DTYPE_BF16:
        return rope_<llaisys::bf16_t>(out, in, pos_ids, theta, seq_len, num_heads, head_dim);
    case LLAISYS_DTYPE_F16:
        return rope_<llaisys::fp16_t>(out, in, pos_ids, theta, seq_len, num_heads, head_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu