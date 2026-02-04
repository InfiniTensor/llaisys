#include "op.hpp"
#include <cmath>

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    //TO_BE_IMPLEMENTED();
    // 获取张量数据指针
    auto out_data = out->data();
    auto in_data = in->data();
    auto weight_data = weight->data();
    
    // 获取维度信息
    auto in_shape = in->shape();
    size_t batch_size = in_shape[0];
    size_t d = in_shape[1];
    auto dtype = in->dtype();
    
    // 根据数据类型进行处理
    switch (static_cast<llaisysDataType_t>(dtype)) {
    case LLAISYS_DTYPE_F32: {
        auto* in_ptr = reinterpret_cast<const float*>(in_data);
        auto* weight_ptr = reinterpret_cast<const float*>(weight_data);
        auto* out_ptr = reinterpret_cast<float*>(out_data);
        
        for (size_t i = 0; i < batch_size; i++) {
            const float* row_in = in_ptr + i * d;
            float* row_out = out_ptr + i * d;
            
            // 计算平方和
            float sum_sq = 0.0f;
            for (size_t j = 0; j < d; j++) {
                sum_sq += row_in[j] * row_in[j];
            }
            
            // 计算 RMS 归一化因子
            float rms = std::sqrt(sum_sq / static_cast<float>(d) + eps);
            float scale = 1.0f / rms;
            
            // 应用归一化和权重
            for (size_t j = 0; j < d; j++) {
                row_out[j] = weight_ptr[j] * (row_in[j] * scale);
            }
        }
        break;
    }
    
    case LLAISYS_DTYPE_F16: {
        auto* in_ptr = reinterpret_cast<const llaisys::fp16_t*>(in_data);
        auto* weight_ptr = reinterpret_cast<const llaisys::fp16_t*>(weight_data);
        auto* out_ptr = reinterpret_cast<llaisys::fp16_t*>(out_data);
        
        for (size_t i = 0; i < batch_size; i++) {
            const llaisys::fp16_t* row_in = in_ptr + i * d;
            llaisys::fp16_t* row_out = out_ptr + i * d;
            
            // 计算平方和（使用float精度）
            float sum_sq = 0.0f;
            for (size_t j = 0; j < d; j++) {
                float val = llaisys::utils::_f16_to_f32(row_in[j]);
                sum_sq += val * val;
            }
            
            // 计算 RMS 归一化因子
            float rms = std::sqrt(sum_sq / static_cast<float>(d) + eps);
            float scale = 1.0f / rms;
            
            // 应用归一化和权重
            for (size_t j = 0; j < d; j++) {
                float val = llaisys::utils::_f16_to_f32(row_in[j]);
                float weight_val = llaisys::utils::_f16_to_f32(weight_ptr[j]);
                row_out[j] = llaisys::utils::_f32_to_f16(weight_val * (val * scale));
            }
        }
        break;
    }
    
    case LLAISYS_DTYPE_BF16: {
        auto* in_ptr = reinterpret_cast<const llaisys::bf16_t*>(in_data);
        auto* weight_ptr = reinterpret_cast<const llaisys::bf16_t*>(weight_data);
        auto* out_ptr = reinterpret_cast<llaisys::bf16_t*>(out_data);
        
        for (size_t i = 0; i < batch_size; i++) {
            const llaisys::bf16_t* row_in = in_ptr + i * d;
            llaisys::bf16_t* row_out = out_ptr + i * d;
            
            // 计算平方和（使用float精度）
            float sum_sq = 0.0f;
            for (size_t j = 0; j < d; j++) {
                float val = llaisys::utils::_bf16_to_f32(row_in[j]);
                sum_sq += val * val;
            }
            
            // 计算 RMS 归一化因子
            float rms = std::sqrt(sum_sq / static_cast<float>(d) + eps);
            float scale = 1.0f / rms;
            
            // 应用归一化和权重
            for (size_t j = 0; j < d; j++) {
                float val = llaisys::utils::_bf16_to_f32(row_in[j]);
                float weight_val = llaisys::utils::_bf16_to_f32(weight_ptr[j]);
                row_out[j] = llaisys::utils::_f32_to_bf16(weight_val * (val * scale));
            }
        }
        break;
    }
    
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(static_cast<llaisysDataType_t>(dtype));
    }
 
}
} // namespace llaisys::ops
