#include "argmax_cpu.hpp"

#include "../../../utils.hpp"
#include "llaisys.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

// cpu侧实现
template <typename T>
void argmax_(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    if (numel == 0) {
        *max_idx = 0;
        // 对于fp16和bf16这种非内置类型，需要用cast转换；其他类型使用默认构造赋0值
        if (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            *max_val = llaisys::utils::cast<T>(0.0f);
        } else {
            *max_val = T{};  
        }
        return;
    }

    T tmp_max_val = vals[0];
    int64_t tmp_max_idx = 0;

    // 对于fp16和bf16，先转为float32进行比较，避免精度丢失
    if constexpr (std::is_same_v<T, llaisys::fp16_t> || std::is_same_v<T, llaisys::bf16_t>) {
        float max_val_float = llaisys::utils::cast<float>(vals[0]);
        for (size_t i = 1; i < numel; ++i) {
            float cur_val_float = llaisys::utils::cast<float>(vals[i]);
            if (cur_val_float > max_val_float) {    
                max_val_float = cur_val_float;
                tmp_max_val = vals[i];
                tmp_max_idx = i;
            }
        }
    } else {
        for (size_t i = 1; i < numel; i++) {
            if (vals[i] > tmp_max_val) {
                tmp_max_val = vals[i];
                tmp_max_idx = i;
            }
        }
    }

    *max_idx = tmp_max_idx;
    *max_val = tmp_max_val;
}

namespace llaisys::ops::cpu {
void argmax(int64_t *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel) {
        // 传入的是std::byte类型的指针，需要转成对应的类型
        switch (type) {
        case LLAISYS_DTYPE_F32:
            return argmax_(max_idx, reinterpret_cast<float *>(max_val),
                           reinterpret_cast<const float*>(vals), numel);
        case LLAISYS_DTYPE_BF16:
            return argmax_(max_idx, reinterpret_cast<llaisys::bf16_t*>(max_val),
                     reinterpret_cast<const llaisys::bf16_t*>(vals), numel);
        case LLAISYS_DTYPE_F16:
            return argmax_(max_idx, reinterpret_cast<llaisys::fp16_t*>(max_val),
                     reinterpret_cast<const llaisys::fp16_t*>(vals), numel);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
        }
    }
} // namespace llaisys::ops::cpu
