#include "swiglu_cpu.hpp"
#include "../../../utils.hpp"

#include <cmath>

namespace {

template <typename T>
inline void swiglu_impl(T *out,
                        const T *gate,
                        const T *up,
                        size_t N, size_t D) {
    const size_t numel = N * D;
    for (size_t i = 0; i < numel; ++i) {
        float g = llaisys::utils::cast<float>(gate[i]);
        float u = llaisys::utils::cast<float>(up[i]);
        float s = 1.0f / (1.0f + std::exp(-g)); // sigmoid(g)
        float y = u * (g * s);                  // up * gate * sigmoid(gate)
        out[i] = llaisys::utils::cast<T>(y);
    }
}

} // anonymous namespace

namespace llaisys::ops::cpu {

void swiglu(std::byte *out,
            const std::byte *gate,
            const std::byte *up,
            llaisysDataType_t type,
            size_t N, size_t D) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return swiglu_impl(reinterpret_cast<float *>(out),
                           reinterpret_cast<const float *>(gate),
                           reinterpret_cast<const float *>(up),
                           N, D);
    case LLAISYS_DTYPE_F16:
        return swiglu_impl(reinterpret_cast<llaisys::fp16_t *>(out),
                           reinterpret_cast<const llaisys::fp16_t *>(gate),
                           reinterpret_cast<const llaisys::fp16_t *>(up),
                           N, D);
    case LLAISYS_DTYPE_BF16:
        return swiglu_impl(reinterpret_cast<llaisys::bf16_t *>(out),
                           reinterpret_cast<const llaisys::bf16_t *>(gate),
                           reinterpret_cast<const llaisys::bf16_t *>(up),
                           N, D);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu
