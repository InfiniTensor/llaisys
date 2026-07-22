#include "op.hpp"
#include <cmath>


namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    //TO_BE_IMPLEMENTED();
    // 获取张量数据指针
    auto out_data = out->data();
    auto in_data = in->data();
    auto pos_ids_data = pos_ids->data();
    
    // 获取维度信息
    auto in_shape = in->shape();
    size_t seqlen = in_shape[0];
    size_t nhead = in_shape[1];
    size_t d = in_shape[2];
    size_t d_half = d / 2;
    auto dtype = in->dtype();
    
    // 转换位置ID数据指针
    auto* pos_ids_ptr = reinterpret_cast<const int64_t*>(pos_ids_data);
    
    // 根据数据类型进行处理
    switch (static_cast<llaisysDataType_t>(dtype)) {
    case LLAISYS_DTYPE_F32: {
        auto* in_ptr = reinterpret_cast<const float*>(in_data);
        auto* out_ptr = reinterpret_cast<float*>(out_data);
        
        for (size_t s = 0; s < seqlen; s++) {
            int64_t pos = pos_ids_ptr[s];
            
            for (size_t h = 0; h < nhead; h++) {
                const float* head_in = in_ptr + (s * nhead + h) * d;
                float* head_out = out_ptr + (s * nhead + h) * d;
                
                for (size_t j = 0; j < d_half; j++) {
                    // 计算角度 phi = pos / (theta^(2j/d))
                    // float exponent = static_cast<float>(2 * j) / static_cast<float>(d);
                    // //float freq = 1.0f / std::pow(theta, exponent);
                    // float freq = std::exp(-exponent * std::log(theta));
                    // float phi = static_cast<float>(pos) * freq;
                    // float exponent = -static_cast<float>(2 * j) / static_cast<float>(d);
                    // float freq = std::pow(theta, exponent);  // 直接计算 θ^(-2j/d)
                    // float phi = static_cast<float>(pos) * freq;
                    float exponent = static_cast<float>(2 * j) / static_cast<float>(d);
                    float freq = std::pow(theta, exponent);  // theta^(2j/d)
                    float phi = static_cast<float>(pos) / freq;  // pos / theta^(2j/d)
                    
                    // 计算 sin 和 cos
                    float cos_phi = std::cos(phi);
                    float sin_phi = std::sin(phi);
                    
                    // 获取 a_j 和 b_j
                    float a_j = head_in[j];
                    float b_j = head_in[d_half + j];
                    
                    // 计算旋转后的值
                    head_out[j] = a_j * cos_phi - b_j * sin_phi;
                    head_out[d_half + j] = b_j * cos_phi + a_j * sin_phi;
                }
            }
        }
        break;
    }
    
    case LLAISYS_DTYPE_F16: {
        auto* in_ptr = reinterpret_cast<const llaisys::fp16_t*>(in_data);
        auto* out_ptr = reinterpret_cast<llaisys::fp16_t*>(out_data);
        
        for (size_t s = 0; s < seqlen; s++) {
            int64_t pos = pos_ids_ptr[s];
            
            for (size_t h = 0; h < nhead; h++) {
                const llaisys::fp16_t* head_in = in_ptr + (s * nhead + h) * d;
                llaisys::fp16_t* head_out = out_ptr + (s * nhead + h) * d;
                
                for (size_t j = 0; j < d_half; j++) {
                    // 计算角度 phi = pos / (theta^(2j/d))
                    // float exponent = static_cast<float>(2 * j) / static_cast<float>(d);
                    // //float freq = 1.0f / std::pow(theta, exponent);
                    // float freq = std::exp(-exponent * std::log(theta));
                    // float phi = static_cast<float>(pos) * freq;
                    // float exponent = -static_cast<float>(2 * j) / static_cast<float>(d);
                    // float freq = std::pow(theta, exponent);  // 直接计算 θ^(-2j/d)
                    // float phi = static_cast<float>(pos) * freq;
                    float exponent = static_cast<float>(2 * j) / static_cast<float>(d);
                    float freq = std::pow(theta, exponent);  // theta^(2j/d)
                    float phi = static_cast<float>(pos) / freq;  // pos / theta^(2j/d)
                    
                    // 计算 sin 和 cos
                    float cos_phi = std::cos(phi);
                    float sin_phi = std::sin(phi);
                    
                    // 获取 a_j 和 b_j
                    float a_j = llaisys::utils::_f16_to_f32(head_in[j]);
                    float b_j = llaisys::utils::_f16_to_f32(head_in[d_half + j]);
                    
                    // 计算旋转后的值
                    float a_j_out = a_j * cos_phi - b_j * sin_phi;
                    float b_j_out = b_j * cos_phi + a_j * sin_phi;
                    
                    head_out[j] = llaisys::utils::_f32_to_f16(a_j_out);
                    head_out[d_half + j] = llaisys::utils::_f32_to_f16(b_j_out);
                }
            }
        }
        break;
    }
    
    case LLAISYS_DTYPE_BF16: {
        auto* in_ptr = reinterpret_cast<const llaisys::bf16_t*>(in_data);
        auto* out_ptr = reinterpret_cast<llaisys::bf16_t*>(out_data);
        
        for (size_t s = 0; s < seqlen; s++) {
            int64_t pos = pos_ids_ptr[s];
            
            for (size_t h = 0; h < nhead; h++) {
                const llaisys::bf16_t* head_in = in_ptr + (s * nhead + h) * d;
                llaisys::bf16_t* head_out = out_ptr + (s * nhead + h) * d;
                
                for (size_t j = 0; j < d_half; j++) {
                    // 计算角度 phi = pos / (theta^(2j/d))
                    // float exponent = static_cast<float>(2 * j) / static_cast<float>(d);
                    // //float freq = 1.0f / std::pow(theta, exponent);
                    // float freq = std::exp(-exponent * std::log(theta));
                    // float phi = static_cast<float>(pos) * freq;
                    // float exponent = -static_cast<float>(2 * j) / static_cast<float>(d);
                    // float freq = std::pow(theta, exponent);  // 直接计算 θ^(-2j/d)
                    // float phi = static_cast<float>(pos) * freq;
                    float exponent = static_cast<float>(2 * j) / static_cast<float>(d);
                    float freq = std::pow(theta, exponent);  // theta^(2j/d)
                    float phi = static_cast<float>(pos) / freq;  // pos / theta^(2j/d)
                    
                    // 计算 sin 和 cos
                    float cos_phi = std::cos(phi);
                    float sin_phi = std::sin(phi);
                    
                    // 获取 a_j 和 b_j
                    float a_j = llaisys::utils::_bf16_to_f32(head_in[j]);
                    float b_j = llaisys::utils::_bf16_to_f32(head_in[d_half + j]);
                    
                    // 计算旋转后的值
                    float a_j_out = a_j * cos_phi - b_j * sin_phi;
                    float b_j_out = b_j * cos_phi + a_j * sin_phi;
                    
                    head_out[j] = llaisys::utils::_f32_to_bf16(a_j_out);
                    head_out[d_half + j] = llaisys::utils::_f32_to_bf16(b_j_out);
                }
            }
        }
        break;
    }
    
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(static_cast<llaisysDataType_t>(dtype));
    }
 
}
} // namespace llaisys::ops
