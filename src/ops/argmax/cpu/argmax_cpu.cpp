#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cstdint>
#include <algorithm>

namespace llaisys::ops::cpu {

template <typename T>
void argmax_(std::byte *max_idx_bytes, std::byte *max_val_bytes, const std::byte *vals_bytes, size_t size) {
    auto max_idx = reinterpret_cast<int64_t *>(max_idx_bytes);
    auto max_val = reinterpret_cast<T *>(max_val_bytes);
    auto vals = reinterpret_cast<const T *>(vals_bytes);

    *max_idx = 0;
    *max_val = vals[0];

    for (size_t i = 1; i < size; ++i) {
        float curr_val = llaisys::utils::cast<float>(vals[i]);
        float current_max = llaisys::utils::cast<float>(*max_val);
        if (curr_val > current_max) {
            *max_idx = static_cast<int64_t>(i);
            *max_val = vals[i];
        }
    }
}

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t size) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_<float>(max_idx, max_val, vals, size);
    case LLAISYS_DTYPE_BF16:
        return argmax_<llaisys::bf16_t>(max_idx, max_val, vals, size);
    case LLAISYS_DTYPE_F16:
        return argmax_<llaisys::fp16_t>(max_idx, max_val, vals, size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu