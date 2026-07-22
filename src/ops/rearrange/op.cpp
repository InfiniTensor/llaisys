#include "op.hpp"

namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in) {
    //TO_BE_IMPLEMENTED();
    // 获取张量数据指针
    auto out_data = out->data();
    auto in_data = in->data();
    
    // 获取总元素数
    size_t total_elements = in->numel();
    auto dtype = in->dtype();
    
    // 根据数据类型进行处理
    switch (static_cast<llaisysDataType_t>(dtype)) {
    case LLAISYS_DTYPE_F32: {
        auto* in_ptr = reinterpret_cast<const float*>(in_data);
        auto* out_ptr = reinterpret_cast<float*>(out_data);
        
        // 简单复制所有元素
        for (size_t i = 0; i < total_elements; i++) {
            out_ptr[i] = in_ptr[i];
        }
        break;
    }
    
    case LLAISYS_DTYPE_F16: {
        auto* in_ptr = reinterpret_cast<const llaisys::fp16_t*>(in_data);
        auto* out_ptr = reinterpret_cast<llaisys::fp16_t*>(out_data);
        
        for (size_t i = 0; i < total_elements; i++) {
            out_ptr[i] = in_ptr[i];
        }
        break;
    }
    
    case LLAISYS_DTYPE_BF16: {
        auto* in_ptr = reinterpret_cast<const llaisys::bf16_t*>(in_data);
        auto* out_ptr = reinterpret_cast<llaisys::bf16_t*>(out_data);
        
        for (size_t i = 0; i < total_elements; i++) {
            out_ptr[i] = in_ptr[i];
        }
        break;
    }
    
    case LLAISYS_DTYPE_I64: {
        auto* in_ptr = reinterpret_cast<const int64_t*>(in_data);
        auto* out_ptr = reinterpret_cast<int64_t*>(out_data);
        
        for (size_t i = 0; i < total_elements; i++) {
            out_ptr[i] = in_ptr[i];
        }
        break;
    }
    
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(static_cast<llaisysDataType_t>(dtype));
    }

}
} // namespace llaisys::ops
