#include "swiglu_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

namespace {
// SwiGLU 激活函数实现：SwiGLU(g, u) = u * g * σ(g)
// 其中 σ(g) = 1 / (1 + exp(-g)) 是 Sigmoid 函数
template <typename T>
void swiglu_impl(std::byte *output,
                 const std::byte *gate_input,
                 const std::byte *up_input,
                 size_t element_count) {
    const T *gate_data = reinterpret_cast<const T *>(gate_input);
    const T *up_data = reinterpret_cast<const T *>(up_input);
    T *out_data = reinterpret_cast<T *>(output);

    for (size_t idx = 0; idx < element_count; ++idx) {
        // 将输入提升到 float 精度进行计算（保证数值稳定性）
        float g_val = llaisys::utils::cast<float>(gate_data[idx]);
        float u_val = llaisys::utils::cast<float>(up_data[idx]);

        // 计算 Sigmoid(g)
        float sigmoid_g = 1.0f / (1.0f + std::exp(-g_val));

        // SwiGLU 输出：u * g * σ(g)
        out_data[idx] = llaisys::utils::cast<T>(u_val * g_val * sigmoid_g);
    }
}
}

namespace llaisys::ops::cpu {

// 在 CPU 上执行 SwiGLU 激活函数
void swiglu(std::byte *output,
            const std::byte *gate,
            const std::byte *up,
            llaisysDataType_t dtype,
            size_t numel) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return swiglu_impl<float>(output, gate, up, numel);
    case LLAISYS_DTYPE_BF16:
        return swiglu_impl<llaisys::bf16_t>(output, gate, up, numel);
    case LLAISYS_DTYPE_F16:
        return swiglu_impl<llaisys::fp16_t>(output, gate, up, numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu