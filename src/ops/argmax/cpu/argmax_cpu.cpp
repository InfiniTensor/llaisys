#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstdint>
#include <limits>

template <typename T>
void argmax_(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    if (numel == 0) {
        return;
    }

    float current_max_f;
    size_t best_idx = 0;

    current_max_f = llaisys::utils::cast<float>(vals[0]);

    for (size_t i = 1; i < numel; i++) {
        float val_f;
        val_f = llaisys::utils::cast<float>(vals[i]);

        if (val_f > current_max_f) {
            current_max_f = val_f;
            best_idx = i;
        }
    }

    *max_idx = static_cast<int64_t>(best_idx);

    *max_val = vals[best_idx];
}

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel) {
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
}