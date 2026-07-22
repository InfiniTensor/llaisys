#include "op.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    //TO_BE_IMPLEMENTED();
        // 获取张量数据指针
    auto out_data = out->data();
    auto index_data = index->data();
    auto weight_data = weight->data();
    
    // 获取维度信息
    auto out_shape = out->shape();
    auto index_shape = index->shape();
    auto weight_shape = weight->shape();
    
    size_t num_indices = index_shape[0];
    size_t embedding_dim = weight_shape[1];
    auto dtype = weight->dtype();
    
    // 转换索引数据指针
    auto* index_ptr = reinterpret_cast<const int64_t*>(index_data);
    
    // 根据数据类型进行处理
    switch (static_cast<llaisysDataType_t>(dtype)) {
    case LLAISYS_DTYPE_F32: {
        auto* weight_ptr = reinterpret_cast<const float*>(weight_data);
        auto* out_ptr = reinterpret_cast<float*>(out_data);
        
        for (size_t i = 0; i < num_indices; i++) {
            int64_t idx = index_ptr[i];
            const float* src_row = weight_ptr + idx * embedding_dim;
            float* dst_row = out_ptr + i * embedding_dim;
            
            for (size_t j = 0; j < embedding_dim; j++) {
                dst_row[j] = src_row[j];
            }
        }
        break;
    }
    
    case LLAISYS_DTYPE_F16: {
        auto* weight_ptr = reinterpret_cast<const llaisys::fp16_t*>(weight_data);
        auto* out_ptr = reinterpret_cast<llaisys::fp16_t*>(out_data);
        
        for (size_t i = 0; i < num_indices; i++) {
            int64_t idx = index_ptr[i];
            const llaisys::fp16_t* src_row = weight_ptr + idx * embedding_dim;
            llaisys::fp16_t* dst_row = out_ptr + i * embedding_dim;
            
            for (size_t j = 0; j < embedding_dim; j++) {
                dst_row[j] = src_row[j];
            }
        }
        break;
    }
    
    case LLAISYS_DTYPE_BF16: {
        auto* weight_ptr = reinterpret_cast<const llaisys::bf16_t*>(weight_data);
        auto* out_ptr = reinterpret_cast<llaisys::bf16_t*>(out_data);
        
        for (size_t i = 0; i < num_indices; i++) {
            int64_t idx = index_ptr[i];
            const llaisys::bf16_t* src_row = weight_ptr + idx * embedding_dim;
            llaisys::bf16_t* dst_row = out_ptr + i * embedding_dim;
            
            for (size_t j = 0; j < embedding_dim; j++) {
                dst_row[j] = src_row[j];
            }
        }
        break;
    }
    
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(static_cast<llaisysDataType_t>(dtype));
    }

}
} // namespace llaisys::ops
