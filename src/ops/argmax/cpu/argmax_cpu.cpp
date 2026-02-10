#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void argmax_(size_t *max_idx, T *max_val, const T* vals, size_t numel){
    size_t max_index = 0;
    float max_value = llaisys::utils::cast<float>(vals[0]);
    
    for(size_t i = 1; i < numel; ++i){
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>){
            max_value = llaisys::utils::cast<float>(max_value);
            float current_value = llaisys::utils::cast<float>(vals[i]);
            if(current_value > max_value){
                max_value = current_value;
                max_index = i;
            }
        } else {
            if(vals[i] > max_value){
                max_value = vals[i];
                max_index = i;
            }
        }
    }

    *max_idx = max_index;
    *max_val = llaisys::utils::cast<T>(max_value);
}

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel){
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_(reinterpret_cast<size_t *>(max_idx), reinterpret_cast<float *>(max_val), reinterpret_cast<const float *>(vals), numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_(reinterpret_cast<size_t *>(max_idx), reinterpret_cast<llaisys::bf16_t *>(max_val), reinterpret_cast<const llaisys::bf16_t *>(vals), numel);
    case LLAISYS_DTYPE_F16:
        return argmax_(reinterpret_cast<size_t *>(max_idx), reinterpret_cast<llaisys::fp16_t *>(max_val), reinterpret_cast<const llaisys::fp16_t *>(vals), numel);
    default:    
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
