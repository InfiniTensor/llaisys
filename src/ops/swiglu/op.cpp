#include "op.hpp"
#include "../../utils.hpp"
#include <cmath> // 用于 std::exp

namespace llaisys::ops {

template <typename T>
void swiglu_cpu(tensor_t out, tensor_t gate, tensor_t up) {
    // 1. 获取元素总数
    // SwiGLU 是 Element-wise (逐元素) 操作，所以不关心形状，只关心总个数
    size_t n = out->numel();

    // 2. 获取数据指针
    T* out_ptr = reinterpret_cast<T*>(out->data());
    const T* gate_ptr = reinterpret_cast<const T*>(gate->data());
    const T* up_ptr = reinterpret_cast<const T*>(up->data());

    // 3. 逐元素计算
    for (size_t i = 0; i < n; ++i) {
        // 读取并转为 float
        float g = utils::cast<float>(gate_ptr[i]); // Gate 值
        float u = utils::cast<float>(up_ptr[i]);   // Up 值

        // --- 计算 SiLU ---
        // SiLU(g) = g / (1 + exp(-g))
        float sigmoid = 1.0f / (1.0f + std::exp(-g));
        float silu = g * sigmoid;

        // --- 计算 SwiGLU ---
        // result = up * SiLU(gate)
        float res = u * silu;

        // 存回结果
        out_ptr[i] = utils::cast<T>(res);
    }
}

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    switch (gate->dtype()) {
        case LLAISYS_DTYPE_F32:
            swiglu_cpu<float>(out, gate, up);
            break;
        case LLAISYS_DTYPE_F16:
            swiglu_cpu<fp16_t>(out, gate, up);
            break;
        case LLAISYS_DTYPE_BF16:
            swiglu_cpu<bf16_t>(out, gate, up);
            break;
        default:
            break;
    }
}

} // namespace llaisys::ops