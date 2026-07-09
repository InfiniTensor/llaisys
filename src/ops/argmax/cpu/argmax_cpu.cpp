#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void argmax_(size_t *max_idx, T *max_vals, const T *vals, size_t numel) {
    size_t max_idx_ = 0;
    T max_val_ = vals[0];
    for (size_t i = 1; i < numel; i++) {
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            if (llaisys::utils::cast<float>(vals[i]) > llaisys::utils::cast<float>(max_val_)) {
                max_idx_ = i;
                max_val_ = llaisys::utils::cast<T>(vals[i]);
            }
        } else {
            if (vals[i] > max_val_) {
                max_idx_ = i;
                max_val_ = vals[i];
            }
        }
    }
    *max_idx = max_idx_;
    *max_vals = max_val_;
}

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_vals, const std::byte *vals, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_(reinterpret_cast<size_t *>(max_idx), reinterpret_cast<float *>(max_vals), reinterpret_cast<const float *>(vals), numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_(reinterpret_cast<size_t *>(max_idx), reinterpret_cast<llaisys::bf16_t *>(max_vals),
                    reinterpret_cast<const llaisys::bf16_t *>(vals), numel);
    case LLAISYS_DTYPE_F16:
        return argmax_(reinterpret_cast<size_t *>(max_idx), reinterpret_cast<llaisys::fp16_t *>(max_vals),
                    reinterpret_cast<const llaisys::fp16_t *>(vals), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu