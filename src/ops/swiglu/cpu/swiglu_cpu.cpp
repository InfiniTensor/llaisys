
#include "swiglu_cpu.hpp"
#include "../../../utils.hpp"

#include <cstddef>
#include <cmath>

namespace {

void swiglu_f32(float *out, const float *gate, const float *up, std::ptrdiff_t numel) {
#pragma omp parallel for schedule(static) if (numel >= 4096)
    for (std::ptrdiff_t i = 0; i < numel; ++i) {
        const float g_val = gate[i];
        out[i] = up[i] * (g_val / (1.0f + std::exp(-g_val)));
    }
}

template <typename T>
void swiglu_generic(T *out, const T *gate, const T *up, std::ptrdiff_t numel) {
#pragma omp parallel for schedule(static) if (numel >= 4096)
    for (std::ptrdiff_t i = 0; i < numel; ++i) {
        const float g_val = llaisys::utils::cast<float>(gate[i]);
        const float u_val = llaisys::utils::cast<float>(up[i]);
        out[i] = llaisys::utils::cast<T>(u_val * (g_val / (1.0f + std::exp(-g_val))));
    }
}

} // namespace

namespace llaisys::ops::cpu {

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, 
            llaisysDataType_t dtype, size_t numel) {
    const auto elem_count = static_cast<std::ptrdiff_t>(numel);
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        swiglu_f32(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(gate), reinterpret_cast<const float *>(up), elem_count);
        break;
    case LLAISYS_DTYPE_BF16:
        swiglu_generic(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(gate), reinterpret_cast<const llaisys::bf16_t *>(up), elem_count);
        break;
    case LLAISYS_DTYPE_F16:
        swiglu_generic(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(gate), reinterpret_cast<const llaisys::fp16_t *>(up), elem_count);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
