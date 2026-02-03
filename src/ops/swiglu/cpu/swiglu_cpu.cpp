#include "swiglu_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

namespace llaisys::ops::cpu {

template <typename T>
static void swiglu_kernel(std::byte* out_raw,
                          const std::byte* gate_raw,
                          const std::byte* up_raw,
                          size_t numel) {
    const T* gate = reinterpret_cast<const T*>(gate_raw);
    const T* up   = reinterpret_cast<const T*>(up_raw);
    T* out        = reinterpret_cast<T*>(out_raw);

    for (size_t i = 0; i < numel; ++i) {
        float g   = llaisys::utils::cast<float>(gate[i]);
        float u   = llaisys::utils::cast<float>(up[i]);
        float sig = g / (1.0f + std::exp(-g));
        out[i] = llaisys::utils::cast<T>(u * sig);
    }
}

void swiglu(std::byte* out,
            const std::byte* gate,
            const std::byte* up,
            llaisysDataType_t dtype,
            size_t numel) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return swiglu_kernel<float>(out, gate, up, numel);
    case LLAISYS_DTYPE_F16:
        return swiglu_kernel<llaisys::fp16_t>(out, gate, up, numel);
    case LLAISYS_DTYPE_BF16:
        return swiglu_kernel<llaisys::bf16_t>(out, gate, up, numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
