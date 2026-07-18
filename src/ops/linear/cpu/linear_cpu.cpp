#include "linear_cpu.hpp"

#include "../../../utils.hpp"
#include <type_traits>

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias,
             size_t batch_size, size_t in_features, size_t out_features) {
    using acc_t = std::conditional_t<
        std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>,
        float, T>;

    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < out_features; ++j) {
            acc_t sum = acc_t{};
            for (size_t k = 0; k < in_features; ++k) {
                if constexpr (std::is_same_v<T, acc_t>) {
                    sum += in[i * in_features + k] * weight[j * in_features + k];
                } else {
                    sum += llaisys::utils::cast<acc_t>(in[i * in_features + k]) * llaisys::utils::cast<acc_t>(weight[j * in_features + k]);
                }
            }
            if (bias != nullptr) {
                if constexpr (std::is_same_v<T, acc_t>) {
                    sum += bias[j];
                } else {
                    sum += llaisys::utils::cast<acc_t>(bias[j]);
                }
            }
            if constexpr (std::is_same_v<T, acc_t>) {
                out[i * out_features + j] = sum;
            } else {
                out[i * out_features + j] = llaisys::utils::cast<T>(sum);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, llaisysDataType_t type, size_t batch_size, size_t in_features, size_t out_features) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(weight), reinterpret_cast<const float *>(bias), batch_size, in_features, out_features);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in), reinterpret_cast<const llaisys::bf16_t *>(weight), reinterpret_cast<const llaisys::bf16_t *>(bias), batch_size, in_features, out_features);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in), reinterpret_cast<const llaisys::fp16_t *>(weight), reinterpret_cast<const llaisys::fp16_t *>(bias), batch_size, in_features, out_features);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
} 
}