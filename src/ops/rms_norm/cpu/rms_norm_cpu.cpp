#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, const float eps, std::vector<size_t> shapes){
    size_t dimi = shapes[0];
    size_t dimj = shapes[1];
    for(size_t i = 0; i < dimi; ++i){
        float sum_sq = 0.0f;
        for(size_t j = 0; j < dimj; ++j){
            float val = llaisys::utils::cast<float>(in[i * dimj + j]);
            sum_sq += val * val;
        }

        float rms = std::sqrt(sum_sq / dimj + eps);
        float inv_rms = 1.0f / rms;
        for(size_t j = 0; j < dimj; ++j){
            float val = llaisys::utils::cast<float>(in[i * dimj + j]);
            float w   = llaisys::utils::cast<float>(weight[j]);
            float res = val * inv_rms * w;
            out[i * dimj + j] = llaisys::utils::cast<T>(res);
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, const float eps, llaisysDataType_t type, std::vector<size_t> shapes){
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(weight), eps, shapes);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in), reinterpret_cast<const llaisys::bf16_t *>(weight), eps, shapes);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in), reinterpret_cast<const llaisys::fp16_t *>(weight), eps, shapes);
    default:    
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
