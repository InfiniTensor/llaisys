#include "swiglu_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstring>

namespace llaisys::ops::cpu {

template <typename T>
void swiglu_(std::byte *out_bytes, const std::byte *gate_bytes, const std::byte *up_bytes, size_t seq_len, size_t intermediate_size) {
    auto out = reinterpret_cast<T *>(out_bytes);
    auto gate = reinterpret_cast<const T *>(gate_bytes);
    auto up = reinterpret_cast<const T *>(up_bytes);

    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < intermediate_size; ++j) {
            size_t index = i * intermediate_size + j;
            float gate_val = llaisys::utils::cast<float>(gate[index]);
            float up_val = llaisys::utils::cast<float>(up[index]);
            
            // Compute gate / (1 + e^{-gate})
            float exp_neg_gate = std::exp(-gate_val);
            float denominator = 1.0f + exp_neg_gate;
            float gate_div = gate_val / denominator;
            
            // Compute out = up * (gate / (1 + e^{-gate}))
            float out_val = up_val * gate_div;
            
            out[index] = llaisys::utils::cast<T>(out_val);
        }
    }
}

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t type, size_t seq_len, size_t intermediate_size) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return swiglu_<float>(out, gate, up, seq_len, intermediate_size);
    case LLAISYS_DTYPE_BF16:
        return swiglu_<llaisys::bf16_t>(out, gate, up, seq_len, intermediate_size);
    case LLAISYS_DTYPE_F16:
        return swiglu_<llaisys::fp16_t>(out, gate, up, seq_len, intermediate_size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu