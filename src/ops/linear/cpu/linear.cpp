#include "linear.hpp"

#include "../../../utils.hpp"

template <typename T>
static void linear_(
    T* out,
    const T* in,
    const T* weight,
    const T* bias,
    size_t batch_size,
    size_t in_features,
    size_t out_features
) {
    using namespace llaisys::utils;
    for (size_t b = 0; b < batch_size; b++) {
        const T* in_batch = in + b * in_features;
        T* out_batch = out + b * out_features;

        for (size_t o = 0; o < out_features; o++) {
            float sum = bias ? cast<float>(bias[o]) : 0.0f;
            const T* weight_ = weight + o * in_features;
            
            for (size_t i = 0; i < in_features; i++) {
                sum += cast<float>(in_batch[i]) * cast<float>(weight_[i]);
            }
            out_batch[o] = cast<T>(sum);
        }
    }
}

namespace llaisys::ops::cpu {
void linear(
    void* out,
    const void* in,
    const void* weight,
    const void* bias,
    size_t batch_size,
    size_t in_features,
    size_t out_features,
    llaisysDataType_t data_type
) {
    switch (data_type) {
    case LLAISYS_DTYPE_F32:
        return linear_(
            reinterpret_cast<float*>(out),
            reinterpret_cast<const float*>(in),
            reinterpret_cast<const float*>(weight),
            reinterpret_cast<const float*>(bias),
            batch_size,
            in_features,
            out_features
        );
    case LLAISYS_DTYPE_F16:
        return linear_(
            reinterpret_cast<llaisys::fp16_t*>(out),
            reinterpret_cast<const llaisys::fp16_t*>(in),
            reinterpret_cast<const llaisys::fp16_t*>(weight),
            reinterpret_cast<const llaisys::fp16_t*>(bias),
            batch_size,
            in_features,
            out_features
        );
    case LLAISYS_DTYPE_BF16:
        return linear_(
            reinterpret_cast<llaisys::bf16_t*>(out),
            reinterpret_cast<const llaisys::bf16_t*>(in),
            reinterpret_cast<const llaisys::bf16_t*>(weight),
            reinterpret_cast<const llaisys::bf16_t*>(bias),
            batch_size,
            in_features,
            out_features
        );
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(data_type);
    }
        
}
}