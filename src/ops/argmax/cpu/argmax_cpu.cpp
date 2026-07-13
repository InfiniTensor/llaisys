#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>

template <typename T, typename N>
void argmax_(N *max_idx, T *max_val, const T *vals, size_t numel) {
    max_idx[0] = 0;
    max_val[0] = vals[0];
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        for (size_t i = 1; i < numel; i++) {
            if (llaisys::utils::cast<float>(vals[i]) > llaisys::utils::cast<float>(max_val[0])) {
                max_val[0] = vals[i];
                max_idx[0] = static_cast<N>(i);
            }
        }
    } else {
        for (size_t i = 1; i < numel; i++) {
            if (vals[i] > max_val[0]) {
                max_val[0] = vals[i];
                max_idx[0] = static_cast<N>(i);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<float *>(max_val), reinterpret_cast<const float *>(vals), numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<llaisys::bf16_t *>(max_val),
                       reinterpret_cast<const llaisys::bf16_t *>(vals), numel);
    case LLAISYS_DTYPE_F16:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<llaisys::fp16_t *>(max_val),
                       reinterpret_cast<const llaisys::fp16_t *>(vals), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
