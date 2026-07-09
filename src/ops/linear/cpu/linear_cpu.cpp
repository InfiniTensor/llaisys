#include "linear_cpu.hpp"

#include "../../../utils.hpp"

#include <cstddef>

namespace llaisys::ops::cpu {

template <typename T>
void linear_impl(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, size_t batch_size, size_t in_features, size_t out_features) {
    const T* in_data = reinterpret_cast<const T*>(in);
    const T* weight_data = reinterpret_cast<const T*>(weight);
    T* out_data = reinterpret_cast<T*>(out);
    
    // 矩阵乘法：Y = xW^T
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            T sum = 0;
            for (size_t k = 0; k < in_features; k++) {
                sum += in_data[i * in_features + k] * weight_data[j * in_features + k];
            }
            out_data[i * out_features + j] = sum;
        }
    }
    
    // 添加偏置
    if (bias != nullptr) {
        const T* bias_data = reinterpret_cast<const T*>(bias);
        for (size_t i = 0; i < batch_size; i++) {
            for (size_t j = 0; j < out_features; j++) {
                out_data[i * out_features + j] += bias_data[j];
            }
        }
    }
}

// 处理F16类型的特化实现
template <>
void linear_impl<llaisys::fp16_t>(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, size_t batch_size, size_t in_features, size_t out_features) {
    const llaisys::fp16_t* in_data = reinterpret_cast<const llaisys::fp16_t*>(in);
    const llaisys::fp16_t* weight_data = reinterpret_cast<const llaisys::fp16_t*>(weight);
    llaisys::fp16_t* out_data = reinterpret_cast<llaisys::fp16_t*>(out);
    
    // 矩阵乘法：Y = xW^T
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            float sum = 0;
            for (size_t k = 0; k < in_features; k++) {
                sum += llaisys::utils::cast<float>(in_data[i * in_features + k]) * llaisys::utils::cast<float>(weight_data[j * in_features + k]);
            }
            out_data[i * out_features + j] = llaisys::utils::cast<llaisys::fp16_t>(sum);
        }
    }
    
    // 添加偏置
    if (bias != nullptr) {
        const llaisys::fp16_t* bias_data = reinterpret_cast<const llaisys::fp16_t*>(bias);
        for (size_t i = 0; i < batch_size; i++) {
            for (size_t j = 0; j < out_features; j++) {
                float val = llaisys::utils::cast<float>(out_data[i * out_features + j]) + llaisys::utils::cast<float>(bias_data[j]);
                out_data[i * out_features + j] = llaisys::utils::cast<llaisys::fp16_t>(val);
            }
        }
    }
}

// 处理BF16类型的特化实现
template <>
void linear_impl<llaisys::bf16_t>(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, size_t batch_size, size_t in_features, size_t out_features) {
    const llaisys::bf16_t* in_data = reinterpret_cast<const llaisys::bf16_t*>(in);
    const llaisys::bf16_t* weight_data = reinterpret_cast<const llaisys::bf16_t*>(weight);
    llaisys::bf16_t* out_data = reinterpret_cast<llaisys::bf16_t*>(out);
    
    // 矩阵乘法：Y = xW^T
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            float sum = 0;
            for (size_t k = 0; k < in_features; k++) {
                sum += llaisys::utils::cast<float>(in_data[i * in_features + k]) * llaisys::utils::cast<float>(weight_data[j * in_features + k]);
            }
            out_data[i * out_features + j] = llaisys::utils::cast<llaisys::bf16_t>(sum);
        }
    }
    
    // 添加偏置
    if (bias != nullptr) {
        const llaisys::bf16_t* bias_data = reinterpret_cast<const llaisys::bf16_t*>(bias);
        for (size_t i = 0; i < batch_size; i++) {
            for (size_t j = 0; j < out_features; j++) {
                float val = llaisys::utils::cast<float>(out_data[i * out_features + j]) + llaisys::utils::cast<float>(bias_data[j]);
                out_data[i * out_features + j] = llaisys::utils::cast<llaisys::bf16_t>(val);
            }
        }
    }
}

void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, llaisysDataType_t type, size_t batch_size, size_t in_features, size_t out_features) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_impl<float>(out, in, weight, bias, batch_size, in_features, out_features);
    case LLAISYS_DTYPE_F64:
        return linear_impl<double>(out, in, weight, bias, batch_size, in_features, out_features);
    case LLAISYS_DTYPE_F16:
        return linear_impl<llaisys::fp16_t>(out, in, weight, bias, batch_size, in_features, out_features);
    case LLAISYS_DTYPE_BF16:
        return linear_impl<llaisys::bf16_t>(out, in, weight, bias, batch_size, in_features, out_features);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu