#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <limits>

template <typename T>
void argmax_(int64_t *max_idx, float *max_val, const T *vals, size_t numel) {
    size_t idx = 0;
    float max_value = -std::numeric_limits<float>::infinity();

    for (size_t i = 0; i < numel; i++) {
        float val;
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            val = llaisys::utils::cast<float>(vals[i]);
        } else {
            val = static_cast<float>(vals[i]);
        }

        if (val > max_value) {
            max_value = val;
            idx = i;
        }
    }

    // max_val is always F32
    max_idx[0] = static_cast<int64_t>(idx);
    max_val[0] = max_value;
}

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel) {
    int64_t *idx_ptr = reinterpret_cast<int64_t *>(max_idx);
    float *val_ptr = reinterpret_cast<float *>(max_val); // always F32

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_(idx_ptr, val_ptr, reinterpret_cast<const float *>(vals), numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_(idx_ptr, val_ptr, reinterpret_cast<const llaisys::bf16_t *>(vals), numel);
    case LLAISYS_DTYPE_F16:
        return argmax_(idx_ptr, val_ptr, reinterpret_cast<const llaisys::fp16_t *>(vals), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
