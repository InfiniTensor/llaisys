#include "swiglu_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

template <typename T>
void swiglu_(T *out, const T *gate, const T *up, size_t size) {
    for (size_t i = 0; i < size; i++) {
        float g = llaisys::utils::cast<float>(gate[i]);
        float u = llaisys::utils::cast<float>(up[i]);
        float silu = g / (1.0f + std::exp(-g));
        out[i] = llaisys::utils::cast<T>(u * silu);
    }
}

namespace llaisys::ops::cpu {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t type, size_t size) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return swiglu_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(gate),
                      reinterpret_cast<const float *>(up), size);
    case LLAISYS_DTYPE_BF16:
        return swiglu_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(gate),
                      reinterpret_cast<const llaisys::bf16_t *>(up), size);
    case LLAISYS_DTYPE_F16:
        return swiglu_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(gate),
                      reinterpret_cast<const llaisys::fp16_t *>(up), size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
}
