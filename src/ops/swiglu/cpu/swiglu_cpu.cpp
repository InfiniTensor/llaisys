#include "swiglu_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

// SwiGLU: out_i = up_i * sigmoid(gate_i)
// 其中 sigmoid(x) = x / (1 + exp(-x))
template <typename T>
void swiglu_(T *out, const T *gate, const T *up, size_t numel) {
    for (size_t i = 0; i < numel; i++) {
        float gate_val, up_val;
        
        // 转换到float精度进行计算
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            gate_val = llaisys::utils::cast<float>(gate[i]);
            up_val = llaisys::utils::cast<float>(up[i]);
        } else {
            gate_val = static_cast<float>(gate[i]);
            up_val = static_cast<float>(up[i]);
        }
        
        // 计算sigmoid(gate_val) = gate_val / (1 + exp(-gate_val))
        // 为了数值稳定性，当gate_val很大时直接使用gate_val
        float sigmoid_val;
        if (gate_val > 20.0f) {
            // 当gate很大时，exp(-gate) ≈ 0，所以sigmoid(gate) ≈ gate/1 = gate
            sigmoid_val = gate_val;
        } else if (gate_val < -20.0f) {
            // 当gate很小时，exp(-gate)很大，所以sigmoid(gate) ≈ 0
            sigmoid_val = 0.0f;
        } else {
            // 正常情况下计算sigmoid
            sigmoid_val = gate_val / (1.0f + std::exp(-gate_val));
        }
        
        // 计算最终结果：out_i = up_i * sigmoid(gate_i)
        float result = up_val * sigmoid_val;
        
        // 转换回目标类型
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            out[i] = llaisys::utils::cast<T>(result);
        } else {
            out[i] = static_cast<T>(result);
        }
    }
}

namespace llaisys::ops::cpu {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, 
           llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return swiglu_(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(gate),
            reinterpret_cast<const float *>(up),
            numel
        );
    case LLAISYS_DTYPE_BF16:
        return swiglu_(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(gate),
            reinterpret_cast<const llaisys::bf16_t *>(up),
            numel
        );
    case LLAISYS_DTYPE_F16:
        return swiglu_(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(gate),
            reinterpret_cast<const llaisys::fp16_t *>(up),
            numel
        );
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
