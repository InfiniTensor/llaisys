#include "swiglu_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void swiglu_(std::byte *out_raw, const std::byte *gate_raw, const std::byte *up_raw, const size_t numel) {
    T *out = reinterpret_cast<T*>(out_raw);
    const T *gate = reinterpret_cast<const T*>(gate_raw);
    const T *up = reinterpret_cast<const T*>(up_raw);
    for (size_t i = 0; i < numel; i++) {
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            out[i] = llaisys::utils::cast<T>(llaisys::utils::cast<float>(up[i]) * llaisys::utils::cast<float>(gate[i]) / (1 + std::exp(- llaisys::utils::cast<float>(gate[i]))));
        } else {
            out[i] = llaisys::utils::cast<T>(static_cast<float>(up[i]) * static_cast<float>(gate[i]) / (1 + std::exp(- static_cast<float>(gate[i]))));
        }
    }
}

#define DISPATCH_SWIGLU(dtype, ctype) case dtype: swiglu_<ctype>(out, gate, up, numel); break;

namespace llaisys::ops::cpu {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t type, const size_t numel) {
    switch (type) {
        DISPATCH_SWIGLU(LLAISYS_DTYPE_F32, float)
        DISPATCH_SWIGLU(LLAISYS_DTYPE_BF16, llaisys::bf16_t)
        DISPATCH_SWIGLU(LLAISYS_DTYPE_F16, llaisys::fp16_t)
        DISPATCH_SWIGLU(LLAISYS_DTYPE_I32, int32_t)
        DISPATCH_SWIGLU(LLAISYS_DTYPE_F64, double)
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
