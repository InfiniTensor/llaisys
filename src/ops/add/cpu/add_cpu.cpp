#include "add_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void add_(std::byte *c, const std::byte *a, const std::byte *b, size_t numel) {
    T *c_typed = reinterpret_cast<T*>(c);
    const T *a_typed = reinterpret_cast<const T*>(a);
    const T *b_typed = reinterpret_cast<const T*>(b);
    for (size_t i = 0; i < numel; i++) {
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            c_typed[i] = llaisys::utils::cast<T>(llaisys::utils::cast<float>(a_typed[i]) + llaisys::utils::cast<float>(b_typed[i]));
        } else {
            c_typed[i] = a_typed[i] + b_typed[i];
        }
    }
}

#define DISPATCH_ADD(dtype, ctype) case dtype: add_<ctype>(c, a, b, numel); break;

namespace llaisys::ops::cpu {
void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel) {
    switch (type) {
        DISPATCH_ADD(LLAISYS_DTYPE_F32, float)
        DISPATCH_ADD(LLAISYS_DTYPE_BF16, llaisys::bf16_t)
        DISPATCH_ADD(LLAISYS_DTYPE_F16, llaisys::fp16_t)
        DISPATCH_ADD(LLAISYS_DTYPE_I32, int32_t)
        DISPATCH_ADD(LLAISYS_DTYPE_F64, double)
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
