#include "op.hpp"

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {

    // 获取张量数据指针
    auto idx_data = max_idx->data();
    auto val_data = max_val->data();
    auto vals_data = vals->data();
    
    // 获取维度信息
    size_t numel = vals->numel();
    auto dtype = vals->dtype();
    
    // 根据数据类型进行处理
    switch (static_cast<llaisysDataType_t>(dtype)) {
    case LLAISYS_DTYPE_F32: {
        auto* vals_ptr = reinterpret_cast<const float*>(vals_data);
        auto* max_val_ptr = reinterpret_cast<float*>(val_data);
        auto* max_idx_ptr = reinterpret_cast<int64_t*>(idx_data);
        
        float max_val = vals_ptr[0];
        int64_t max_idx_val = 0;
        
        for (size_t i = 1; i < numel; i++) {
            if (vals_ptr[i] > max_val) {
                max_val = vals_ptr[i];
                max_idx_val = i;
            }
        }
        
        *max_val_ptr = max_val;
        *max_idx_ptr = max_idx_val;
        break;
    }
    
    case LLAISYS_DTYPE_F16: {
        auto* vals_ptr = reinterpret_cast<const llaisys::fp16_t*>(vals_data);
        auto* max_val_ptr = reinterpret_cast<llaisys::fp16_t*>(val_data);
        auto* max_idx_ptr = reinterpret_cast<int64_t*>(idx_data);
        
        float max_val = llaisys::utils::_f16_to_f32(vals_ptr[0]);
        int64_t max_idx_val = 0;
        
        for (size_t i = 1; i < numel; i++) {
            float curr_val = llaisys::utils::_f16_to_f32(vals_ptr[i]);
            if (curr_val > max_val) {
                max_val = curr_val;
                max_idx_val = i;
            }
        }
        
        *max_val_ptr = llaisys::utils::_f32_to_f16(max_val);
        *max_idx_ptr = max_idx_val;
        break;
    }
    
    case LLAISYS_DTYPE_BF16: {
        auto* vals_ptr = reinterpret_cast<const llaisys::bf16_t*>(vals_data);
        auto* max_val_ptr = reinterpret_cast<llaisys::bf16_t*>(val_data);
        auto* max_idx_ptr = reinterpret_cast<int64_t*>(idx_data);
        
        float max_val = llaisys::utils::_bf16_to_f32(vals_ptr[0]);
        int64_t max_idx_val = 0;
        
        for (size_t i = 1; i < numel; i++) {
            float curr_val = llaisys::utils::_bf16_to_f32(vals_ptr[i]);
            if (curr_val > max_val) {
                max_val = curr_val;
                max_idx_val = i;
            }
        }
        
        *max_val_ptr = llaisys::utils::_f32_to_bf16(max_val);
        *max_idx_ptr = max_idx_val;
        break;
    }
    
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(static_cast<llaisysDataType_t>(dtype));
    }

}
} // namespace llaisys::ops
