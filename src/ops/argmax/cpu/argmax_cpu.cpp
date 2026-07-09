#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cstdint>

template <typename T>
void argmax_(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    if (numel == 0) {
        return;
    }
    size_t idx = 0;
    T max = vals[0];
    for (size_t i = 1; i < numel; i++) {
        if constexpr (std::is_same_v<T, llaisys::bf16_t> ||
                      std::is_same_v<T, llaisys::fp16_t>) {
            float vf = llaisys::utils::cast<float>(vals[i]);
            float bf = llaisys::utils::cast<float>(max);
            if (vf > bf) {
                idx = i;
                max = vals[i];
            }
        } else {
            if (vals[i] > max) {
                idx = i;
                max = vals[i];
            }
        }
    }
    *max_idx = static_cast<int64_t>(idx);
    *max_val = max;
}

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
            llaisysDataType_t vals_dtype, size_t numel) {
    int64_t *max_idx_ptr = reinterpret_cast<int64_t *>(max_idx);
    switch (vals_dtype) {
    case LLAISYS_DTYPE_F32:
        argmax_(max_idx_ptr, reinterpret_cast<float *>(max_val),
                reinterpret_cast<const float *>(vals), numel);
        return;
    case LLAISYS_DTYPE_BF16:
        argmax_(max_idx_ptr, reinterpret_cast<llaisys::bf16_t *>(max_val),
                reinterpret_cast<const llaisys::bf16_t *>(vals), numel);
        return;
    case LLAISYS_DTYPE_F16:
        argmax_(max_idx_ptr, reinterpret_cast<llaisys::fp16_t *>(max_val),
                reinterpret_cast<const llaisys::fp16_t *>(vals), numel);
        return;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(vals_dtype);
    }
}
} // namespace llaisys::ops::cpu
