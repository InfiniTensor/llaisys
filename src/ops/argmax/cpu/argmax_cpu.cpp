#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cstddef>

namespace llaisys::ops::cpu {

template <typename T>
void argmax_impl(std::byte *max_idx, std::byte *max_val, const std::byte *vals, size_t size) {
    const T* vals_data = reinterpret_cast<const T*>(vals);
    T* max_val_data = reinterpret_cast<T*>(max_val);
    int64_t* max_idx_data = reinterpret_cast<int64_t*>(max_idx);
    
    // 初始化最大值和索引
    T max_val_val = vals_data[0];
    size_t max_idx_val = 0;
    
    // 遍历所有元素找最大值
    for (size_t i = 1; i < size; i++) {
        if (vals_data[i] > max_val_val) {
            max_val_val = vals_data[i];
            max_idx_val = i;
        }
    }
    
    // 存储结果
    max_val_data[0] = max_val_val;
    max_idx_data[0] = static_cast<int64_t>(max_idx_val);
}

// 处理F16类型的特化实现
template <>
void argmax_impl<llaisys::fp16_t>(std::byte *max_idx, std::byte *max_val, const std::byte *vals, size_t size) {
    const llaisys::fp16_t* vals_data = reinterpret_cast<const llaisys::fp16_t*>(vals);
    llaisys::fp16_t* max_val_data = reinterpret_cast<llaisys::fp16_t*>(max_val);
    int64_t* max_idx_data = reinterpret_cast<int64_t*>(max_idx);
    
    // 初始化最大值和索引
    float max_val_val = llaisys::utils::cast<float>(vals_data[0]);
    size_t max_idx_val = 0;
    
    // 遍历所有元素找最大值
    for (size_t i = 1; i < size; i++) {
        float current_val = llaisys::utils::cast<float>(vals_data[i]);
        if (current_val > max_val_val) {
            max_val_val = current_val;
            max_idx_val = i;
        }
    }
    
    // 存储结果
    max_val_data[0] = llaisys::utils::cast<llaisys::fp16_t>(max_val_val);
    max_idx_data[0] = static_cast<int64_t>(max_idx_val);
}

// 处理BF16类型的特化实现
template <>
void argmax_impl<llaisys::bf16_t>(std::byte *max_idx, std::byte *max_val, const std::byte *vals, size_t size) {
    const llaisys::bf16_t* vals_data = reinterpret_cast<const llaisys::bf16_t*>(vals);
    llaisys::bf16_t* max_val_data = reinterpret_cast<llaisys::bf16_t*>(max_val);
    int64_t* max_idx_data = reinterpret_cast<int64_t*>(max_idx);
    
    // 初始化最大值和索引
    float max_val_val = llaisys::utils::cast<float>(vals_data[0]);
    size_t max_idx_val = 0;
    
    // 遍历所有元素找最大值
    for (size_t i = 1; i < size; i++) {
        float current_val = llaisys::utils::cast<float>(vals_data[i]);
        if (current_val > max_val_val) {
            max_val_val = current_val;
            max_idx_val = i;
        }
    }
    
    // 存储结果
    max_val_data[0] = llaisys::utils::cast<llaisys::bf16_t>(max_val_val);
    max_idx_data[0] = static_cast<int64_t>(max_idx_val);
}

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t size) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_impl<float>(max_idx, max_val, vals, size);
    case LLAISYS_DTYPE_F64:
        return argmax_impl<double>(max_idx, max_val, vals, size);
    case LLAISYS_DTYPE_I8:
        return argmax_impl<int8_t>(max_idx, max_val, vals, size);
    case LLAISYS_DTYPE_I16:
        return argmax_impl<int16_t>(max_idx, max_val, vals, size);
    case LLAISYS_DTYPE_I32:
        return argmax_impl<int32_t>(max_idx, max_val, vals, size);
    case LLAISYS_DTYPE_I64:
        return argmax_impl<int64_t>(max_idx, max_val, vals, size);
    case LLAISYS_DTYPE_U8:
        return argmax_impl<uint8_t>(max_idx, max_val, vals, size);
    case LLAISYS_DTYPE_U16:
        return argmax_impl<uint16_t>(max_idx, max_val, vals, size);
    case LLAISYS_DTYPE_U32:
        return argmax_impl<uint32_t>(max_idx, max_val, vals, size);
    case LLAISYS_DTYPE_U64:
        return argmax_impl<uint64_t>(max_idx, max_val, vals, size);
    case LLAISYS_DTYPE_F16:
        return argmax_impl<llaisys::fp16_t>(max_idx, max_val, vals, size);
    case LLAISYS_DTYPE_BF16:
        return argmax_impl<llaisys::bf16_t>(max_idx, max_val, vals, size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu