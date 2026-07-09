#include "swiglu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>

template <typename T>
void swiglu_(
    T *output,
    const T *input,
    const T *gate,
    size_t total_size
) {
    for(size_t i = 0; i < total_size; i++) {
        float gate_val = llaisys::utils::cast<float>(gate[i]);
        float silu = gate_val / (1.0f + std::exp(-gate_val));
        float input_val = llaisys::utils::cast<float>(input[i]);
        output[i] = llaisys::utils::cast<T>(input_val * silu);
    }
}

namespace llaisys::ops::cpu {
void swiglu(
    std::byte *output,
    const std::byte *input,
    const std::byte *gate,
    llaisysDataType_t type,
    size_t total_size
) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return swiglu_(reinterpret_cast<float *>(output), reinterpret_cast<const float *>(input),
                       reinterpret_cast<const float *>(gate), total_size);
    case LLAISYS_DTYPE_F16:
        return swiglu_(reinterpret_cast<llaisys::fp16_t *>(output), reinterpret_cast<const llaisys::fp16_t *>(input),
                       reinterpret_cast<const llaisys::fp16_t *>(gate), total_size);
    case LLAISYS_DTYPE_BF16:
        return swiglu_(reinterpret_cast<llaisys::bf16_t *>(output), reinterpret_cast<const llaisys::bf16_t *>(input),
                       reinterpret_cast<const llaisys::bf16_t *>(gate), total_size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu

