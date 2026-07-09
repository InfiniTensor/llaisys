#include "argmax_cpu.hpp"
#include "../../../utils.hpp"
#include <cstdint>

template <typename T>
void argmax_(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    if (numel == 0) return;

    // 初始化：假设第一个元素就是最大的
    T current_max = vals[0];
    int64_t current_idx = 0;

    for (size_t i = 1; i < numel; i++) {
        // 使用 cast 转为 float 比较，以支持 fp16/bf16
        if (llaisys::utils::cast<float>(vals[i]) > llaisys::utils::cast<float>(current_max)) {
            current_max = vals[i];
            current_idx = static_cast<int64_t>(i);
        }
    }

    *max_val = current_max;
    *max_idx = current_idx;
}

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, 
            llaisysDataType_t type, size_t numel) {
    // max_idx 固定为 I64
    int64_t *idx_ptr = reinterpret_cast<int64_t *>(max_idx);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_(idx_ptr, reinterpret_cast<float *>(max_val), 
                       reinterpret_cast<const float *>(vals), numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_(idx_ptr, reinterpret_cast<llaisys::bf16_t *>(max_val),
                       reinterpret_cast<const llaisys::bf16_t *>(vals), numel);
    case LLAISYS_DTYPE_F16:
        return argmax_(idx_ptr, reinterpret_cast<llaisys::fp16_t *>(max_val),
                       reinterpret_cast<const llaisys::fp16_t *>(vals), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu