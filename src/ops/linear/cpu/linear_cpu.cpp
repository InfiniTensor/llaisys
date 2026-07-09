#include "linear_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>

// Y = X * W^T + b
// X: [batch_size, in_features]
// W: [out_features, in_features] (需要转置)
// b: [out_features] (可选)
// Y: [batch_size, out_features]
template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias,
             size_t batch_size, size_t in_features, size_t out_features) {
    // 对于每个批次样本
    for (size_t b = 0; b < batch_size; b++) {
        // 对于每个输出特征
        for (size_t o = 0; o < out_features; o++) {
            float sum = 0.0f;

            // 计算点积: X[b, :] * W[o, :]^T
            for (size_t i = 0; i < in_features; i++) {
                float x_val, w_val;

                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    x_val = llaisys::utils::cast<float>(in[b * in_features + i]);
                    w_val = llaisys::utils::cast<float>(weight[o * in_features + i]);
                } else {
                    x_val = static_cast<float>(in[b * in_features + i]);
                    w_val = static_cast<float>(weight[o * in_features + i]);
                }

                sum += x_val * w_val;
            }

            // 添加偏置（如果提供）
            if (bias != nullptr) {
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    sum += llaisys::utils::cast<float>(bias[o]);
                } else {
                    sum += static_cast<float>(bias[o]);
                }
            }

            // 存储结果
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                out[b * out_features + o] = llaisys::utils::cast<T>(sum);
            } else {
                out[b * out_features + o] = static_cast<T>(sum);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t batch_size, size_t in_features, size_t out_features) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                       reinterpret_cast<const float *>(weight),
                       bias ? reinterpret_cast<const float *>(bias) : nullptr,
                       batch_size, in_features, out_features);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                       reinterpret_cast<const llaisys::bf16_t *>(weight),
                       bias ? reinterpret_cast<const llaisys::bf16_t *>(bias) : nullptr,
                       batch_size, in_features, out_features);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                       reinterpret_cast<const llaisys::fp16_t *>(weight),
                       bias ? reinterpret_cast<const llaisys::fp16_t *>(bias) : nullptr,
                       batch_size, in_features, out_features);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
