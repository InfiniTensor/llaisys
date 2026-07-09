#include "swiglu_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

template <typename T>
void swiglu_(T *out, const T *gate, const T *up, size_t numel) {
    // out[i] = up[i] * (gate[i] / (1 + exp(-gate[i])))
    // This is: up[i] * gate[i] * sigmoid(gate[i])
    for (size_t i = 0; i < numel; i++) {
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            float g = llaisys::utils::cast<float>(gate[i]);
            float u = llaisys::utils::cast<float>(up[i]);
            float sigmoid_g = 1.0f / (1.0f + std::exp(-g));
            out[i] = llaisys::utils::cast<T>(u * g * sigmoid_g);
        } else {
            float g = static_cast<float>(gate[i]);
            float u = static_cast<float>(up[i]);
            float sigmoid_g = 1.0f / (1.0f + std::exp(-g));
            out[i] = static_cast<T>(u * g * sigmoid_g);
        }
    }
}

namespace llaisys::ops::cpu {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t dtype, size_t numel) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return swiglu_<float>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(gate),
            reinterpret_cast<const float *>(up),
            numel);
    case LLAISYS_DTYPE_BF16:
        return swiglu_<llaisys::bf16_t>(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(gate),
            reinterpret_cast<const llaisys::bf16_t *>(up),
            numel);
    case LLAISYS_DTYPE_F16:
        return swiglu_<llaisys::fp16_t>(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(gate),
            reinterpret_cast<const llaisys::fp16_t *>(up),
            numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
