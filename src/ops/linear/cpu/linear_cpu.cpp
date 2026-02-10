#include "linear_cpu.hpp"

#include "../../../utils.hpp"

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias, std::vector<size_t> shapes){
    size_t dimi = shapes[0];
    size_t dimk = shapes[1];
    size_t dimj = shapes[2];
    for(size_t i = 0; i < dimi; ++i){
        for(size_t j = 0; j < dimj; ++j){
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>){
                float sum = 0.0f;
                for(size_t k = 0; k < dimk; ++k){
                    sum = sum + llaisys::utils::cast<float>(in[i * dimk + k]) * llaisys::utils::cast<float>(weight[j * dimk + k]);
                }
                if(bias != nullptr){
                    sum = sum + llaisys::utils::cast<float>(bias[j]);
                }
                out[i * dimj + j] = llaisys::utils::cast<T>(sum);
            } else {
                T sum = 0.0f;
                for(size_t k = 0; k < dimk; ++k){
                    sum = sum + in[i * dimk + k] * weight[j * dimk + k];
                }
                if(bias != nullptr){
                    sum = sum + bias[j];
                }
                out[i * dimj + j] = sum;
            }
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, llaisysDataType_t type, std::vector<size_t> shapes){
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(weight), bias ? reinterpret_cast<const float *>(bias) : nullptr, shapes);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in), reinterpret_cast<const llaisys::bf16_t *>(weight), bias ? reinterpret_cast<const llaisys::bf16_t *>(bias) : nullptr, shapes);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in), reinterpret_cast<const llaisys::fp16_t *>(weight), bias ? reinterpret_cast<const llaisys::fp16_t *>(bias) : nullptr, shapes);
    default:    
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
