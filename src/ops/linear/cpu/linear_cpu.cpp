#include "linear_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstring>

namespace llaisys::ops::cpu {

template <typename T>
void linear_(std::byte *out_bytes, const std::byte *in_bytes, const std::byte *weight_bytes, const std::byte *bias_bytes, size_t batch_size, size_t in_dim, size_t out_dim) {
    auto out = reinterpret_cast<T *>(out_bytes);
    auto in = reinterpret_cast<const T *>(in_bytes);
    auto weight = reinterpret_cast<const T *>(weight_bytes);
    auto bias = reinterpret_cast<const T *>(bias_bytes);

    // Initialize output with bias if provided
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < out_dim; ++j) {
            if (bias) {
                out[i * out_dim + j] = bias[j];
            } else {
                out[i * out_dim + j] = llaisys::utils::cast<T>(0.0f);
            }
        }
    }

    // Compute Y = xW^T + b
    if constexpr (std::is_same_v<T, float>) {
        // For float32, compute directly without casting
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t k = 0; k < in_dim; ++k) {
                float x = in[i * in_dim + k];
                for (size_t j = 0; j < out_dim; ++j) {
                    float w = weight[j * in_dim + k];
                    out[i * out_dim + j] += x * w;
                }
            }
        }
    } else {
        // For float16 and bfloat16, compute in float32 for better precision
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < out_dim; ++j) {
                float sum = llaisys::utils::cast<float>(out[i * out_dim + j]);
                for (size_t k = 0; k < in_dim; ++k) {
                    float x = llaisys::utils::cast<float>(in[i * in_dim + k]);
                    float w = llaisys::utils::cast<float>(weight[j * in_dim + k]);
                    sum += x * w;
                }
                out[i * out_dim + j] = llaisys::utils::cast<T>(sum);
            }
        }
    }
}

void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, llaisysDataType_t type, size_t batch_size, size_t in_dim, size_t out_dim) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_<float>(out, in, weight, bias, batch_size, in_dim, out_dim);
    case LLAISYS_DTYPE_BF16:
        return linear_<llaisys::bf16_t>(out, in, weight, bias, batch_size, in_dim, out_dim);
    case LLAISYS_DTYPE_F16:
        return linear_<llaisys::fp16_t>(out, in, weight, bias, batch_size, in_dim, out_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu