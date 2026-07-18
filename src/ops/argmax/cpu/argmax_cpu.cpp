#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <type_traits>

template <typename T>
void argmax_(int64_t *max_idx, T *max_val, const T *vals, size_t size) {
    auto max = vals[0];
    size_t max_index = 0;
    for (size_t i = 1; i < size; i++) {
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            if (llaisys::utils::cast<float>(vals[i]) > llaisys::utils::cast<float>(max)) {
                max = vals[i];
                max_index = i;
            }
        } else {
            if (vals[i] > max) {
                max = vals[i];
                max_index = i;
            }
        }
    }
    max_val[0] = max;
    max_idx[0] = static_cast<int64_t>(max_index); 
}

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t size) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<float *>(max_val),
                        reinterpret_cast<const float *>(vals), size);
    case LLAISYS_DTYPE_BF16:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<llaisys::bf16_t *>(max_val),
                        reinterpret_cast<const llaisys::bf16_t *>(vals), size);
    case LLAISYS_DTYPE_F16:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<llaisys::fp16_t *>(max_val),
                        reinterpret_cast<const llaisys::fp16_t *>(vals), size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
