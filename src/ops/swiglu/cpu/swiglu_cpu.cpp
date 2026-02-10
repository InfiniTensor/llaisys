
#include "swiglu_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

namespace {

template <typename T>
void swiglu_(T *out, const T *gate, const T *up, size_t numel) {
    for (size_t i = 0; i < numel; ++i) {
        float g_val = llaisys::utils::cast<float>(gate[i]);
        float u_val = llaisys::utils::cast<float>(up[i]);
        
        float swish_g = g_val / (1.0f + std::exp(-g_val));
        float res = u_val * swish_g;
        out[i] = llaisys::utils::cast<T>(res);
    }
}

} // namespace

namespace llaisys::ops::cpu {

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, 
            llaisysDataType_t dtype, size_t numel) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        swiglu_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(gate), reinterpret_cast<const float *>(up), numel);
        break;
    case LLAISYS_DTYPE_BF16:
        swiglu_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(gate), reinterpret_cast<const llaisys::bf16_t *>(up), numel);
        break;
    case LLAISYS_DTYPE_F16:
        swiglu_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(gate), reinterpret_cast<const llaisys::fp16_t *>(up), numel);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu