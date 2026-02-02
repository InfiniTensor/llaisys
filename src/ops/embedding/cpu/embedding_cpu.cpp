#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cstddef>

namespace llaisys::ops::cpu {

template <typename T>
void embedding_impl(std::byte *out, const std::byte *index, const std::byte *weight, size_t out_size, size_t weight_dim1) {
    const int64_t* index_data = reinterpret_cast<const int64_t*>(index);
    const T* weight_data = reinterpret_cast<const T*>(weight);
    T* out_data = reinterpret_cast<T*>(out);
    
    size_t batch_size = out_size / weight_dim1;
    
    for (size_t i = 0; i < batch_size; i++) {
        int64_t idx = index_data[i];
        // 处理负索引
        if (idx < 0) {
            idx = weight_dim1 + idx;
        }
        
        // 复制对应的行
        for (size_t j = 0; j < weight_dim1; j++) {
            out_data[i * weight_dim1 + j] = weight_data[idx * weight_dim1 + j];
        }
    }
}

// 处理F16类型的特化实现
template <>
void embedding_impl<llaisys::fp16_t>(std::byte *out, const std::byte *index, const std::byte *weight, size_t out_size, size_t weight_dim1) {
    const int64_t* index_data = reinterpret_cast<const int64_t*>(index);
    const llaisys::fp16_t* weight_data = reinterpret_cast<const llaisys::fp16_t*>(weight);
    llaisys::fp16_t* out_data = reinterpret_cast<llaisys::fp16_t*>(out);
    
    size_t batch_size = out_size / weight_dim1;
    
    for (size_t i = 0; i < batch_size; i++) {
        int64_t idx = index_data[i];
        // 处理负索引
        if (idx < 0) {
            idx = weight_dim1 + idx;
        }
        
        // 复制对应的行
        for (size_t j = 0; j < weight_dim1; j++) {
            out_data[i * weight_dim1 + j] = weight_data[idx * weight_dim1 + j];
        }
    }
}

// 处理BF16类型的特化实现
template <>
void embedding_impl<llaisys::bf16_t>(std::byte *out, const std::byte *index, const std::byte *weight, size_t out_size, size_t weight_dim1) {
    const int64_t* index_data = reinterpret_cast<const int64_t*>(index);
    const llaisys::bf16_t* weight_data = reinterpret_cast<const llaisys::bf16_t*>(weight);
    llaisys::bf16_t* out_data = reinterpret_cast<llaisys::bf16_t*>(out);
    
    size_t batch_size = out_size / weight_dim1;
    
    for (size_t i = 0; i < batch_size; i++) {
        int64_t idx = index_data[i];
        // 处理负索引
        if (idx < 0) {
            idx = weight_dim1 + idx;
        }
        
        // 复制对应的行
        for (size_t j = 0; j < weight_dim1; j++) {
            out_data[i * weight_dim1 + j] = weight_data[idx * weight_dim1 + j];
        }
    }
}

void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t type, size_t out_size, size_t weight_dim1) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_impl<float>(out, index, weight, out_size, weight_dim1);
    case LLAISYS_DTYPE_F64:
        return embedding_impl<double>(out, index, weight, out_size, weight_dim1);
    case LLAISYS_DTYPE_F16:
        return embedding_impl<llaisys::fp16_t>(out, index, weight, out_size, weight_dim1);
    case LLAISYS_DTYPE_BF16:
        return embedding_impl<llaisys::bf16_t>(out, index, weight, out_size, weight_dim1);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu