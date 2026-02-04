#include "op.hpp"
#include <cmath>


namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
     //TO_BE_IMPLEMENTED();
    // 获取张量数据指针
    auto out_data = out->data();
    auto gate_data = gate->data();
    auto up_data = up->data();
    
    // 获取维度信息
    auto gate_shape = gate->shape();
    size_t seqlen = gate_shape[0];
    size_t intermediate_size = gate_shape[1];
    auto dtype = gate->dtype();
    
    // 根据数据类型进行处理
    switch (static_cast<llaisysDataType_t>(dtype)) {
    case LLAISYS_DTYPE_F32: {
        auto* gate_ptr = reinterpret_cast<const float*>(gate_data);
        auto* up_ptr = reinterpret_cast<const float*>(up_data);
        auto* out_ptr = reinterpret_cast<float*>(out_data);
        
        for (size_t i = 0; i < seqlen; i++) {
            const float* gate_row = gate_ptr + i * intermediate_size;
            const float* up_row = up_ptr + i * intermediate_size;
            float* out_row = out_ptr + i * intermediate_size;
            
            for (size_t j = 0; j < intermediate_size; j++) {
                // // SwiGLU: out = up * sigmoid(gate)
                // float gate_val = gate_row[j];
                // float sigmoid_gate = 1.0f / (1.0f + std::exp(-gate_val));
                // out_row[j] = up_row[j] * sigmoid_gate;
                // 正确的 SwiGLU 公式：out = up * SiLU(gate) = up * (gate * sigmoid(gate))
                float gate_val = gate_row[j];
                float sigmoid_gate = 1.0f / (1.0f + std::exp(-gate_val));
                float silu_gate = gate_val * sigmoid_gate;  // SiLU 激活函数
                out_row[j] = up_row[j] * silu_gate;
            }
        }
        break;
    }
    
    case LLAISYS_DTYPE_F16: {
        auto* gate_ptr = reinterpret_cast<const llaisys::fp16_t*>(gate_data);
        auto* up_ptr = reinterpret_cast<const llaisys::fp16_t*>(up_data);
        auto* out_ptr = reinterpret_cast<llaisys::fp16_t*>(out_data);
        
        for (size_t i = 0; i < seqlen; i++) {
            const llaisys::fp16_t* gate_row = gate_ptr + i * intermediate_size;
            const llaisys::fp16_t* up_row = up_ptr + i * intermediate_size;
            llaisys::fp16_t* out_row = out_ptr + i * intermediate_size;
            
            for (size_t j = 0; j < intermediate_size; j++) {
                // float gate_val = llaisys::utils::_f16_to_f32(gate_row[j]);
                // float sigmoid_gate = 1.0f / (1.0f + std::exp(-gate_val));
                // float up_val = llaisys::utils::_f16_to_f32(up_row[j]);
                // float out_val = up_val * sigmoid_gate;
                // out_row[j] = llaisys::utils::_f32_to_f16(out_val);
                float gate_val = llaisys::utils::_f16_to_f32(gate_row[j]);
                float sigmoid_gate = 1.0f / (1.0f + std::exp(-gate_val));
                float silu_gate = gate_val * sigmoid_gate;  // SiLU 激活函数
                float up_val = llaisys::utils::_f16_to_f32(up_row[j]);
                float out_val = up_val * silu_gate;
                out_row[j] = llaisys::utils::_f32_to_f16(out_val);
            }
        }
        break;
    }
    
    case LLAISYS_DTYPE_BF16: {
        auto* gate_ptr = reinterpret_cast<const llaisys::bf16_t*>(gate_data);
        auto* up_ptr = reinterpret_cast<const llaisys::bf16_t*>(up_data);
        auto* out_ptr = reinterpret_cast<llaisys::bf16_t*>(out_data);
        
        for (size_t i = 0; i < seqlen; i++) {
            const llaisys::bf16_t* gate_row = gate_ptr + i * intermediate_size;
            const llaisys::bf16_t* up_row = up_ptr + i * intermediate_size;
            llaisys::bf16_t* out_row = out_ptr + i * intermediate_size;
            
            for (size_t j = 0; j < intermediate_size; j++) {
                // float gate_val = llaisys::utils::_bf16_to_f32(gate_row[j]);
                // float sigmoid_gate = 1.0f / (1.0f + std::exp(-gate_val));
                // float up_val = llaisys::utils::_bf16_to_f32(up_row[j]);
                // float out_val = up_val * sigmoid_gate;
                // out_row[j] = llaisys::utils::_f32_to_bf16(out_val);
                float gate_val = llaisys::utils::_bf16_to_f32(gate_row[j]);
                float sigmoid_gate = 1.0f / (1.0f + std::exp(-gate_val));
                float silu_gate = gate_val * sigmoid_gate;  // SiLU 激活函数
                float up_val = llaisys::utils::_bf16_to_f32(up_row[j]);
                float out_val = up_val * silu_gate;
                out_row[j] = llaisys::utils::_f32_to_bf16(out_val);
            }
        }
        break;
    }
    
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(static_cast<llaisysDataType_t>(dtype));
    }
 
}
} // namespace llaisys::ops
