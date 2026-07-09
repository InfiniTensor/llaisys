#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cstdint>

template <typename T>
static void argmax_(std::byte *max_idx, std::byte *max_val, const std::byte *vals, size_t numel) {
    size_t max_i = 0;
    const T *v = reinterpret_cast<const T *>(vals);
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        float max_f = llaisys::utils::cast<float>(v[0]);
        for (size_t i = 1; i < numel; ++i) {
            float cur = llaisys::utils::cast<float>(v[i]);
            if (cur > max_f) {
                max_i = i;
                max_f = cur;
            }
        }
        *reinterpret_cast<T *>(max_val) = llaisys::utils::cast<T>(max_f);
    } else {
        T max_v = v[0];
        for (size_t i = 1; i < numel; ++i) {
            if (v[i] > max_v) {
                max_i = i;
                max_v = v[i];
            }
        }
        *reinterpret_cast<T *>(max_val) = max_v;
    }
    *reinterpret_cast<int64_t *>(max_idx) = static_cast<int64_t>(max_i);
}

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_<float>(max_idx, max_val, vals, numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_<llaisys::bf16_t>(max_idx, max_val, vals, numel);
    case LLAISYS_DTYPE_F16:
        return argmax_<llaisys::fp16_t>(max_idx, max_val, vals, numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
}