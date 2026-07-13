#include "swiglu_cpu.hpp"
#include "../../../utils.hpp"
#include "llaisys.h"
#include <cmath>
#include <omp.h>

template <typename T> static void swiglu_impl(T *out, const T *gate, const T *up, size_t numel) {

#pragma omp parallel for
    for (size_t i = 0; i < numel; ++i) {
        float gate_val = casting(float, gate[i]);
        float up_val = casting(float, up[i]);
        // Swiglu activation: out = up * sigmoid(gate)
        float sigmoid_gate = gate_val / (1.0f + std::exp(-gate_val));
        out[i] = casting(T, up_val * sigmoid_gate);
    }
}

namespace llaisys::ops::cpu {

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, size_t numel, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return swiglu_impl(recast(float *, out), recast(const float *, gate), recast(const float *, up), numel);
    case LLAISYS_DTYPE_F16:
        return swiglu_impl(recast(llaisys::fp16_t *, out), recast(const llaisys::fp16_t *, gate),
                           recast(const llaisys::fp16_t *, up), numel);
    case LLAISYS_DTYPE_BF16:
        return swiglu_impl(recast(llaisys::bf16_t *, out), recast(const llaisys::bf16_t *, gate),
                           recast(const llaisys::bf16_t *, up), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu