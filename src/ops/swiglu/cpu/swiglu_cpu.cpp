#include "swiglu_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstddef>

namespace llaisys::ops::cpu {

// Sigmoid函数模板
template <typename T>
T sigmoid(T x) {
    return static_cast<T>(1.0f / (1.0f + std::exp(-static_cast<float>(x))));
}

// F16类型的sigmoid特化
template <>
llaisys::fp16_t sigmoid<llaisys::fp16_t>(llaisys::fp16_t x) {
    float val = llaisys::utils::cast<float>(x);
    float sigmoid_val = 1.0f / (1.0f + std::exp(-val));
    return llaisys::utils::cast<llaisys::fp16_t>(sigmoid_val);
}

// BF16类型的sigmoid特化
template <>
llaisys::bf16_t sigmoid<llaisys::bf16_t>(llaisys::bf16_t x) {
    float val = llaisys::utils::cast<float>(x);
    float sigmoid_val = 1.0f / (1.0f + std::exp(-val));
    return llaisys::utils::cast<llaisys::bf16_t>(sigmoid_val);
}

template <typename T>
void swiglu_impl(std::byte *out, const std::byte *gate, const std::byte *up, size_t size) {
    const T* gate_data = reinterpret_cast<const T*>(gate);
    const T* up_data = reinterpret_cast<const T*>(up);
    T* out_data = reinterpret_cast<T*>(out);
    
    for (size_t i = 0; i < size; i++) {
        T gate_val = gate_data[i];
        T up_val = up_data[i];
        T sigmoid_gate = sigmoid(gate_val);
        out_data[i] = up_val * gate_val * sigmoid_gate;
    }
}

// F16类型的特化实现
template <>
void swiglu_impl<llaisys::fp16_t>(std::byte *out, const std::byte *gate, const std::byte *up, size_t size) {
    const llaisys::fp16_t* gate_data = reinterpret_cast<const llaisys::fp16_t*>(gate);
    const llaisys::fp16_t* up_data = reinterpret_cast<const llaisys::fp16_t*>(up);
    llaisys::fp16_t* out_data = reinterpret_cast<llaisys::fp16_t*>(out);
    
    for (size_t i = 0; i < size; i++) {
        llaisys::fp16_t gate_val = gate_data[i];
        llaisys::fp16_t up_val = up_data[i];
        llaisys::fp16_t sigmoid_gate = sigmoid(gate_val);
        float up_float = llaisys::utils::cast<float>(up_val);
        float gate_float = llaisys::utils::cast<float>(gate_val);
        float sigmoid_float = llaisys::utils::cast<float>(sigmoid_gate);
        out_data[i] = llaisys::utils::cast<llaisys::fp16_t>(up_float * gate_float * sigmoid_float);
    }
}

// BF16类型的特化实现
template <>
void swiglu_impl<llaisys::bf16_t>(std::byte *out, const std::byte *gate, const std::byte *up, size_t size) {
    const llaisys::bf16_t* gate_data = reinterpret_cast<const llaisys::bf16_t*>(gate);
    const llaisys::bf16_t* up_data = reinterpret_cast<const llaisys::bf16_t*>(up);
    llaisys::bf16_t* out_data = reinterpret_cast<llaisys::bf16_t*>(out);
    
    for (size_t i = 0; i < size; i++) {
        llaisys::bf16_t gate_val = gate_data[i];
        llaisys::bf16_t up_val = up_data[i];
        llaisys::bf16_t sigmoid_gate = sigmoid(gate_val);
        float up_float = llaisys::utils::cast<float>(up_val);
        float gate_float = llaisys::utils::cast<float>(gate_val);
        float sigmoid_float = llaisys::utils::cast<float>(sigmoid_gate);
        out_data[i] = llaisys::utils::cast<llaisys::bf16_t>(up_float * gate_float * sigmoid_float);
    }
}

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t type, size_t size) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return swiglu_impl<float>(out, gate, up, size);
    case LLAISYS_DTYPE_F64:
        return swiglu_impl<double>(out, gate, up, size);
    case LLAISYS_DTYPE_F16:
        return swiglu_impl<llaisys::fp16_t>(out, gate, up, size);
    case LLAISYS_DTYPE_BF16:
        return swiglu_impl<llaisys::bf16_t>(out, gate, up, size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu