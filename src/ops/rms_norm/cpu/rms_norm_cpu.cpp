#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstddef>

namespace llaisys::ops::cpu {

template <typename T>
void rms_norm_impl(std::byte *out, const std::byte *in, const std::byte *weight, float eps, size_t batch_size, size_t hidden_size) {
    const T* in_data = reinterpret_cast<const T*>(in);
    const T* weight_data = reinterpret_cast<const T*>(weight);
    T* out_data = reinterpret_cast<T*>(out);
    
    for (size_t i = 0; i < batch_size; i++) {
        // 计算平方和
        float sum_sq = 0.0f;
        for (size_t j = 0; j < hidden_size; j++) {
            float val = static_cast<float>(in_data[i * hidden_size + j]);
            sum_sq += val * val;
        }
        
        // 计算RMS
        float rms = std::sqrt(sum_sq / static_cast<float>(hidden_size) + eps);
        
        // 计算输出
        for (size_t j = 0; j < hidden_size; j++) {
            float val = static_cast<float>(in_data[i * hidden_size + j]);
            float w = static_cast<float>(weight_data[j]);
            out_data[i * hidden_size + j] = static_cast<T>((w * val) / rms);
        }
    }
}

// 处理F16类型的特化实现
template <>
void rms_norm_impl<llaisys::fp16_t>(std::byte *out, const std::byte *in, const std::byte *weight, float eps, size_t batch_size, size_t hidden_size) {
    const llaisys::fp16_t* in_data = reinterpret_cast<const llaisys::fp16_t*>(in);
    const llaisys::fp16_t* weight_data = reinterpret_cast<const llaisys::fp16_t*>(weight);
    llaisys::fp16_t* out_data = reinterpret_cast<llaisys::fp16_t*>(out);
    
    for (size_t i = 0; i < batch_size; i++) {
        // 计算平方和
        float sum_sq = 0.0f;
        for (size_t j = 0; j < hidden_size; j++) {
            float val = llaisys::utils::cast<float>(in_data[i * hidden_size + j]);
            sum_sq += val * val;
        }
        
        // 计算RMS
        float rms = std::sqrt(sum_sq / static_cast<float>(hidden_size) + eps);
        
        // 计算输出
        for (size_t j = 0; j < hidden_size; j++) {
            float val = llaisys::utils::cast<float>(in_data[i * hidden_size + j]);
            float w = llaisys::utils::cast<float>(weight_data[j]);
            out_data[i * hidden_size + j] = llaisys::utils::cast<llaisys::fp16_t>((w * val) / rms);
        }
    }
}

// 处理BF16类型的特化实现
template <>
void rms_norm_impl<llaisys::bf16_t>(std::byte *out, const std::byte *in, const std::byte *weight, float eps, size_t batch_size, size_t hidden_size) {
    const llaisys::bf16_t* in_data = reinterpret_cast<const llaisys::bf16_t*>(in);
    const llaisys::bf16_t* weight_data = reinterpret_cast<const llaisys::bf16_t*>(weight);
    llaisys::bf16_t* out_data = reinterpret_cast<llaisys::bf16_t*>(out);
    
    for (size_t i = 0; i < batch_size; i++) {
        // 计算平方和
        float sum_sq = 0.0f;
        for (size_t j = 0; j < hidden_size; j++) {
            float val = llaisys::utils::cast<float>(in_data[i * hidden_size + j]);
            sum_sq += val * val;
        }
        
        // 计算RMS
        float rms = std::sqrt(sum_sq / static_cast<float>(hidden_size) + eps);
        
        // 计算输出
        for (size_t j = 0; j < hidden_size; j++) {
            float val = llaisys::utils::cast<float>(in_data[i * hidden_size + j]);
            float w = llaisys::utils::cast<float>(weight_data[j]);
            out_data[i * hidden_size + j] = llaisys::utils::cast<llaisys::bf16_t>((w * val) / rms);
        }
    }
}

void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, float eps, llaisysDataType_t type, size_t batch_size, size_t hidden_size) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_impl<float>(out, in, weight, eps, batch_size, hidden_size);
    case LLAISYS_DTYPE_F64:
        return rms_norm_impl<double>(out, in, weight, eps, batch_size, hidden_size);
    case LLAISYS_DTYPE_F16:
        return rms_norm_impl<llaisys::fp16_t>(out, in, weight, eps, batch_size, hidden_size);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_impl<llaisys::bf16_t>(out, in, weight, eps, batch_size, hidden_size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu