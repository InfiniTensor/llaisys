#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstring>

namespace llaisys::ops::cpu {

template <typename T>
void rms_norm_(std::byte *out_bytes, const std::byte *in_bytes, const std::byte *weight_bytes, float eps, size_t batch_size, size_t hidden_dim) {
    auto out = reinterpret_cast<T *>(out_bytes);
    auto in = reinterpret_cast<const T *>(in_bytes);
    auto weight = reinterpret_cast<const T *>(weight_bytes);

    for (size_t i = 0; i < batch_size; ++i) {
        // Compute sum of squares
        float sum_sq = 0.0f;
        for (size_t j = 0; j < hidden_dim; ++j) {
            float x = llaisys::utils::cast<float>(in[i * hidden_dim + j]);
            sum_sq += x * x;
        }

        // Compute RMS
        float rms = std::sqrt(sum_sq / hidden_dim + eps);

        // Normalize and apply weight
        for (size_t j = 0; j < hidden_dim; ++j) {
            float x = llaisys::utils::cast<float>(in[i * hidden_dim + j]);
            float w = llaisys::utils::cast<float>(weight[j]);
            float y = (x / rms) * w;
            out[i * hidden_dim + j] = llaisys::utils::cast<T>(y);
        }
    }
}

void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, llaisysDataType_t type, float eps, size_t batch_size, size_t hidden_dim) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_<float>(out, in, weight, eps, batch_size, hidden_dim);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_<llaisys::bf16_t>(out, in, weight, eps, batch_size, hidden_dim);
    case LLAISYS_DTYPE_F16:
        return rms_norm_<llaisys::fp16_t>(out, in, weight, eps, batch_size, hidden_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu