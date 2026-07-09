#include "swiglu_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

template <typename T>
void swiglu_(T *out, const T *gate, const T *up, size_t num_elements) {
    for (size_t i = 0; i < num_elements; ++i) {
        float g = llaisys::utils::cast<float>(gate[i]);
        float u = llaisys::utils::cast<float>(up[i]);

        // SiLU(gate) = gate / (1 + exp(-gate))
        float silu_gate = g / (1.0f + std::exp(-g));
        
        // out = up * SiLU(gate)
        out[i] = llaisys::utils::cast<T>(u * silu_gate);
    }
}

namespace llaisys::ops::cpu {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, 
            llaisysDataType_t type, size_t num_elements) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return swiglu_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(gate), 
                       reinterpret_cast<const float *>(up), num_elements);
    case LLAISYS_DTYPE_BF16:
        return swiglu_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(gate), 
                       reinterpret_cast<const llaisys::bf16_t *>(up), num_elements);
    case LLAISYS_DTYPE_F16:
        return swiglu_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(gate), 
                       reinterpret_cast<const llaisys::fp16_t *>(up), num_elements);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
}