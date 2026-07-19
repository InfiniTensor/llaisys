#include "op.hpp"
#include <cmath>

namespace llaisys::ops {

template<typename T>
void swiglu_kernel(tensor_t out, tensor_t gate, tensor_t up){
    T* out_ptr = reinterpret_cast<T*>(out->data());
    const T* gate_ptr = reinterpret_cast<const T*>(gate->data());
    const T* up_ptr = reinterpret_cast<const T*>(up->data());
    size_t n = out->numel();

    for(size_t i = 0; i < n; i++){
        float up_val = utils::cast<float>(up_ptr[i]);
        float gate_val = utils::cast<float>(gate_ptr[i]);
        
        // Swish / SiLU: x / (1 + exp(-x))
        float t_val = gate_val / (1.0f + std::exp(-gate_val));
        
        float out_val = up_val * t_val;
        out_ptr[i] = utils::cast<T>(out_val);
    }
}

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    switch (out->dtype()) {
        case LLAISYS_DTYPE_F16:
            swiglu_kernel<llaisys::fp16_t>(out,gate,up);
            break;
        case LLAISYS_DTYPE_BF16:
            swiglu_kernel<llaisys::bf16_t>(out,gate,up);
            break;
        case LLAISYS_DTYPE_F32:
            swiglu_kernel<float>(out,gate,up);
            break;
        default:
            throw std::runtime_error("Not support this dtype!");
    }
}
} // namespace llaisys::ops