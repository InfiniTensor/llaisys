#include "argmax_cpu.hpp"
#include "../../../utils.hpp"
#include <limits>

template <typename T>
void argmax_(int64_t *idx, T *val, const T *vals, size_t n) {
    float max = -std::numeric_limits<float>::infinity();
    size_t pos = 0;
    for (size_t i = 0; i < n; i++) {
        float v = llaisys::utils::cast<float>(vals[i]);
        if (v > max) { max = v; pos = i; }
    }
    idx[0] = pos;
    val[0] = llaisys::utils::cast<T>(max);
}

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t n) {
    auto idx = reinterpret_cast<int64_t *>(max_idx);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_(idx, reinterpret_cast<float *>(max_val), reinterpret_cast<const float *>(vals), n);
    case LLAISYS_DTYPE_BF16:
        return argmax_(idx, reinterpret_cast<llaisys::bf16_t *>(max_val), reinterpret_cast<const llaisys::bf16_t *>(vals), n);
    case LLAISYS_DTYPE_F16:
        return argmax_(idx, reinterpret_cast<llaisys::fp16_t *>(max_val), reinterpret_cast<const llaisys::fp16_t *>(vals), n);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
}
