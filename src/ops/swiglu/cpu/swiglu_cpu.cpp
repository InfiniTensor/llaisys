#include "swiglu_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath> // std::exp

template <typename T>
void swiglu_kernel(const T *gate, const T *up, T *out, size_t numel) {

    for (size_t i = 0; i < numel; i++) {

        float g_val = llaisys::utils::cast<float>(gate[i]);
        float u_val = llaisys::utils::cast<float>(up[i]);

        float silu_val = g_val / (1.0f + std::exp(-g_val));

        float res = u_val * silu_val;

        out[i] = llaisys::utils::cast<T>(res);
    }
}

namespace llaisys::ops::cpu {

void swiglu(const std::byte *gate_data, const std::byte *up_data, std::byte *out_data,
            llaisysDataType_t dtype, size_t numel) {

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        swiglu_kernel<float>(
            reinterpret_cast<const float *>(gate_data),
            reinterpret_cast<const float *>(up_data),
            reinterpret_cast<float *>(out_data),
            numel);
        break;
    case LLAISYS_DTYPE_F16:
        swiglu_kernel<llaisys::fp16_t>(
            reinterpret_cast<const llaisys::fp16_t *>(gate_data),
            reinterpret_cast<const llaisys::fp16_t *>(up_data),
            reinterpret_cast<llaisys::fp16_t *>(out_data),
            numel);
        break;
    case LLAISYS_DTYPE_BF16:
        swiglu_kernel<llaisys::bf16_t>(
            reinterpret_cast<const llaisys::bf16_t *>(gate_data),
            reinterpret_cast<const llaisys::bf16_t *>(up_data),
            reinterpret_cast<llaisys::bf16_t *>(out_data),
            numel);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu