#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void argmax_(std::byte *max_idx, std::byte *max_val, const std::byte *vals, size_t numel) {
    int64_t *max_idx_typed = reinterpret_cast<int64_t*>(max_idx);
    T *max_val_typed = reinterpret_cast<T*>(max_val);
    const T *vals_typed = reinterpret_cast<const T*>(vals);
    
    *max_idx_typed = 0;
    *max_val_typed = vals_typed[0];
    for (size_t i = 0; i < numel; i++) {
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            float temp = llaisys::utils::cast<float>(vals_typed[i]);
            if (temp > llaisys::utils::cast<float>(*max_val_typed)) {
                *max_idx_typed = static_cast<int64_t>(i);
                *max_val_typed = llaisys::utils::cast<T>(temp);
            }
        } else {
            T temp = vals_typed[i];
            if (temp > *max_val_typed) {
                *max_idx_typed = static_cast<int64_t>(i);
                *max_val_typed = temp;
            }
        }
    }
}

#define DISPATCH_ARGMAX(dtype, ctype) case dtype: argmax_<ctype>(max_idx, max_val, vals, numel); break;

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel) {
    switch (type) {
        DISPATCH_ARGMAX(LLAISYS_DTYPE_F32, float)
        DISPATCH_ARGMAX(LLAISYS_DTYPE_BF16, llaisys::bf16_t)
        DISPATCH_ARGMAX(LLAISYS_DTYPE_F16, llaisys::fp16_t)
        DISPATCH_ARGMAX(LLAISYS_DTYPE_I32, int32_t)
        DISPATCH_ARGMAX(LLAISYS_DTYPE_F64, double)
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
