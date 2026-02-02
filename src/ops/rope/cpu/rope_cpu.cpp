#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstddef>

namespace llaisys::ops::cpu {

template <typename T>
void rope_impl(std::byte *out, const std::byte *in, const std::byte *pos_ids, float theta, size_t seqlen, size_t nhead, size_t d) {
    const T* in_data = reinterpret_cast<const T*>(in);
    const int64_t* pos_ids_data = reinterpret_cast<const int64_t*>(pos_ids);
    T* out_data = reinterpret_cast<T*>(out);
    
    size_t d_half = d / 2;
    
    for (size_t i = 0; i < seqlen; i++) {
        int64_t p = pos_ids_data[i];
        
        for (size_t j = 0; j < nhead; j++) {
            for (size_t k = 0; k < d_half; k++) {
                // 计算角度: phi = p / theta^(2k/d)
                // 使用与PyTorch相同的计算顺序以确保数值一致性
                float exp_value = 2.0f * static_cast<float>(k) / static_cast<float>(d);
                float theta_exp = std::pow(theta, exp_value);
                float phi = static_cast<float>(p) / theta_exp;
                float cos_phi = std::cos(phi);
                float sin_phi = std::sin(phi);
                
                // 获取输入值
                size_t idx = i * nhead * d + j * d + k;
                size_t idx_b = idx + d_half;
                float a = static_cast<float>(in_data[idx]);
                float b = static_cast<float>(in_data[idx_b]);
                
                // 计算输出
                out_data[idx] = static_cast<T>(a * cos_phi - b * sin_phi);
                out_data[idx_b] = static_cast<T>(b * cos_phi + a * sin_phi);
            }
        }
    }
}

// 处理F16类型的特化实现
template <>
void rope_impl<llaisys::fp16_t>(std::byte *out, const std::byte *in, const std::byte *pos_ids, float theta, size_t seqlen, size_t nhead, size_t d) {
    const llaisys::fp16_t* in_data = reinterpret_cast<const llaisys::fp16_t*>(in);
    const int64_t* pos_ids_data = reinterpret_cast<const int64_t*>(pos_ids);
    llaisys::fp16_t* out_data = reinterpret_cast<llaisys::fp16_t*>(out);
    
    size_t d_half = d / 2;
    
    for (size_t i = 0; i < seqlen; i++) {
        int64_t p = pos_ids_data[i];
        
        for (size_t j = 0; j < nhead; j++) {
            for (size_t k = 0; k < d_half; k++) {
                // 计算角度: phi = p / theta^(2k/d)
                // 使用与PyTorch相同的计算顺序以确保数值一致性
                float exp_value = 2.0f * static_cast<float>(k) / static_cast<float>(d);
                float theta_exp = std::pow(theta, exp_value);
                float phi = static_cast<float>(p) / theta_exp;
                float cos_phi = std::cos(phi);
                float sin_phi = std::sin(phi);
                
                // 获取输入值
                size_t idx = i * nhead * d + j * d + k;
                size_t idx_b = idx + d_half;
                float a = llaisys::utils::cast<float>(in_data[idx]);
                float b = llaisys::utils::cast<float>(in_data[idx_b]);
                
                // 计算输出
                out_data[idx] = llaisys::utils::cast<llaisys::fp16_t>(a * cos_phi - b * sin_phi);
                out_data[idx_b] = llaisys::utils::cast<llaisys::fp16_t>(b * cos_phi + a * sin_phi);
            }
        }
    }
}

// 处理BF16类型的特化实现
template <>
void rope_impl<llaisys::bf16_t>(std::byte *out, const std::byte *in, const std::byte *pos_ids, float theta, size_t seqlen, size_t nhead, size_t d) {
    const llaisys::bf16_t* in_data = reinterpret_cast<const llaisys::bf16_t*>(in);
    const int64_t* pos_ids_data = reinterpret_cast<const int64_t*>(pos_ids);
    llaisys::bf16_t* out_data = reinterpret_cast<llaisys::bf16_t*>(out);
    
    size_t d_half = d / 2;
    
    for (size_t i = 0; i < seqlen; i++) {
        int64_t p = pos_ids_data[i];
        
        for (size_t j = 0; j < nhead; j++) {
            for (size_t k = 0; k < d_half; k++) {
                // 计算角度: phi = p / theta^(2k/d)
                // 使用与PyTorch相同的计算顺序以确保数值一致性
                float exp_value = 2.0f * static_cast<float>(k) / static_cast<float>(d);
                float theta_exp = std::pow(theta, exp_value);
                float phi = static_cast<float>(p) / theta_exp;
                float cos_phi = std::cos(phi);
                float sin_phi = std::sin(phi);
                
                // 获取输入值
                size_t idx = i * nhead * d + j * d + k;
                size_t idx_b = idx + d_half;
                float a = llaisys::utils::cast<float>(in_data[idx]);
                float b = llaisys::utils::cast<float>(in_data[idx_b]);
                
                // 计算输出
                out_data[idx] = llaisys::utils::cast<llaisys::bf16_t>(a * cos_phi - b * sin_phi);
                out_data[idx_b] = llaisys::utils::cast<llaisys::bf16_t>(b * cos_phi + a * sin_phi);
            }
        }
    }
}

void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, float theta, llaisysDataType_t type, size_t seqlen, size_t nhead, size_t d) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_impl<float>(out, in, pos_ids, theta, seqlen, nhead, d);
    case LLAISYS_DTYPE_F64:
        return rope_impl<double>(out, in, pos_ids, theta, seqlen, nhead, d);
    case LLAISYS_DTYPE_F16:
        return rope_impl<llaisys::fp16_t>(out, in, pos_ids, theta, seqlen, nhead, d);
    case LLAISYS_DTYPE_BF16:
        return rope_impl<llaisys::bf16_t>(out, in, pos_ids, theta, seqlen, nhead, d);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu