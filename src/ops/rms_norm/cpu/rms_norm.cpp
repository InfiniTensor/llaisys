#include "rms_norm.hpp"

#include "../../../utils.hpp"

#include<algorithm>
#include <cmath>

template <typename T>
void rms_norm_(
    T *output,
    const T *input,
    const T *weight,
    const T *bias,
    size_t batch_size,
    size_t feature_size,
    float epsilon
) {
    for (size_t b = 0; b < batch_size; b++) {
        const T *input_batch = input + b * feature_size;
        T *output_batch = output + b * feature_size;

        // Compute mean square
        float mean_square = 0.0f;
        for (size_t i = 0; i < feature_size; i++) {
            float val = llaisys::utils::cast<float>(input_batch[i]);
            mean_square += val * val;
        }
        mean_square /= static_cast<float>(feature_size);

        // Compute RMS
        float rms = std::sqrt(mean_square + epsilon);

        // Normalize and apply weight and bias
        for (size_t i = 0; i < feature_size; i++) {
            float normalized = llaisys::utils::cast<float>(input_batch[i]) / rms;
            if (weight) {
                normalized *= llaisys::utils::cast<float>(weight[i]);
            }
            if (bias) {
                normalized += llaisys::utils::cast<float>(bias[i]);
            }
            output_batch[i] = llaisys::utils::cast<T>(normalized);
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(
    std::byte *output,
    const std::byte *input,
    const std::byte *weight,
    const std::byte *bias,
    llaisysDataType_t type,
    size_t batch_size,
    size_t feature_size,
    float epsilon
) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(
            reinterpret_cast<float *>(output),
            reinterpret_cast<const float *>(input),
            reinterpret_cast<const float *>(weight),
            reinterpret_cast<const float *>(bias),
            batch_size,
            feature_size,
            epsilon
        );
    case LLAISYS_DTYPE_F16:
        return rms_norm_(
            reinterpret_cast<llaisys::fp16_t *>(output),
            reinterpret_cast<const llaisys::fp16_t *>(input),
            reinterpret_cast<const llaisys::fp16_t *>(weight),
            reinterpret_cast<const llaisys::fp16_t *>(bias),
            batch_size,
            feature_size,
            epsilon
        );
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(
            reinterpret_cast<llaisys::bf16_t *>(output),
            reinterpret_cast<const llaisys::bf16_t *>(input),
            reinterpret_cast<const llaisys::bf16_t *>(weight),
            reinterpret_cast<const llaisys::bf16_t *>(bias),
            batch_size,
            feature_size,
            epsilon
        );
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
}
