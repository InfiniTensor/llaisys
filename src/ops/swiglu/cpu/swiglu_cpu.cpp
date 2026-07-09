#include "swiglu_cpu.hpp"
#include <cmath>
#include <cstdint>
#include "../../../utils.hpp"

template <typename T>
void swiglu_(T *out, const T *gate, const T *up, size_t seq_len, size_t inter_size)
{
    // for each line in up
    for(size_t i = 0; i < seq_len; i++)
    {
        // cal i line base
        T *out_ = out + i * inter_size;
        const T* up_ = up + i * inter_size;
        const T* gate_ = gate + i * inter_size;
        // for each element in the line
        for(size_t j = 0; j < inter_size; j++)
        {
            if constexpr (std::is_same_v<T, llaisys::bf16_t> 
                || std::is_same_v<T, llaisys::fp16_t>) {
                    out_[j] = llaisys::utils::cast<T>(
                        llaisys::utils::cast<float>(up_[j]) * 
                        llaisys::utils::cast<float>(gate_[j]) / 
                        (1.0f + std::exp(-llaisys::utils::cast<float>(gate_[j]))));
                }
            else
                out_[j] = up_[j] * gate_[j] / (1.0f + std::exp(-gate_[j]));
        }
        
    }
}

namespace llaisys::ops::cpu {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t type, size_t seq_len, size_t inter_size) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return swiglu_<float>(reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(gate),
            reinterpret_cast<const float *>(up),
            seq_len, inter_size);
    case LLAISYS_DTYPE_BF16:
        return swiglu_<llaisys::bf16_t>(reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(gate),
            reinterpret_cast<const llaisys::bf16_t *>(up),
            seq_len, inter_size);
    case LLAISYS_DTYPE_F16:
        return swiglu_<llaisys::fp16_t>(reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(gate),
            reinterpret_cast<const llaisys::fp16_t *>(up),
            seq_len, inter_size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
}