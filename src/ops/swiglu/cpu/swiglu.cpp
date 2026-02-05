#include "swiglu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>

template <typename T>
void swiglu_(
    T *output,
    const T *input,
    const T *gate,
    size_t batch_size,
    size_t feature_size
) {
    for (size_t b = 0; b < batch_size; b++) {
        const T *input_batch = input + b * feature_size;
        const T *gate_batch = gate + b * feature_size;
        T *output_batch = output + b * feature_size;
        for (size_t i = 0; i < feature_size; i++) {
            float gate_val = 1.0f / (1.0f + std::exp(-llaisys::utils::cast<float>(gate_batch[i])));
            float input_val = llaisys::utils::cast<float>(input_batch[i]);
            output_batch[i] = llaisys::utils::cast<T>(input_val * gate_val);
        }
    }
}

namespace llaisys::ops::cpu {
void swiglu(
    std::byte *output,
    const std::byte *input,
    const std::byte *gate,
    llaisysDataType_t type,
    size_t batch_size,
    size_t feature_size
) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return swiglu_(reinterpret_cast<float *>(output), reinterpret_cast<const float *>(input),
                       reinterpret_cast<const float *>(gate), batch_size, feature_size);
    case LLAISYS_DTYPE_F16:
        return swiglu_(reinterpret_cast<llaisys::fp16_t *>(output), reinterpret_cast<const llaisys::fp16_t *>(input),
                       reinterpret_cast<const llaisys::fp16_t *>(gate), batch_size, feature_size);
    case LLAISYS_DTYPE_BF16:
        return swiglu_(reinterpret_cast<llaisys::bf16_t *>(output), reinterpret_cast<const llaisys::bf16_t *>(input),
                       reinterpret_cast<const llaisys::bf16_t *>(gate), batch_size, feature_size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu

