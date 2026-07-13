#include "swiglu_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

// SwiGLU: out = up * silu(gate)
// where silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
// Formula: out_i = up_i * gate_i / (1 + exp(-gate_i))

template <typename T>
void swiglu_(T *out, const T *gate, const T *up, size_t numel) {
    for (size_t i = 0; i < numel; ++i) {
        // Convert to double for numerical stability
        double gate_val = llaisys::utils::cast<double>(gate[i]);
        double up_val = llaisys::utils::cast<double>(up[i]);
        
        // silu(gate) = gate / (1 + exp(-gate))
        double silu_gate = gate_val / (1.0 + std::exp(-gate_val));
        
        // out = up * silu(gate)
        double out_val = up_val * silu_gate;
        
        out[i] = llaisys::utils::cast<T>(out_val);
    }
}

namespace llaisys::ops::cpu {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return swiglu_(reinterpret_cast<float *>(out),
                       reinterpret_cast<const float *>(gate),
                       reinterpret_cast<const float *>(up),
                       numel);
    case LLAISYS_DTYPE_BF16:
        return swiglu_(reinterpret_cast<llaisys::bf16_t *>(out),
                       reinterpret_cast<const llaisys::bf16_t *>(gate),
                       reinterpret_cast<const llaisys::bf16_t *>(up),
                       numel);
    case LLAISYS_DTYPE_F16:
        return swiglu_(reinterpret_cast<llaisys::fp16_t *>(out),
                       reinterpret_cast<const llaisys::fp16_t *>(gate),
                       reinterpret_cast<const llaisys::fp16_t *>(up),
                       numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
