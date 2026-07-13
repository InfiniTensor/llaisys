#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, size_t rows, size_t dim, float eps) {
    // Loop over each row
    for (size_t i = 0; i < rows; ++i) {
        const T* in_row = in + i * dim;
        T* out_row = out + i * dim;
        float sum_sq = 0.0f;
        for (size_t j = 0; j < dim; ++j) {
            float val;
            val = llaisys::utils::cast<float>(in_row[j]);
            sum_sq += val * val;
        }

        float mean_sq = sum_sq / static_cast<float>(dim);
        float rms = std::sqrt(mean_sq + eps);
        float inv_rms = 1.0f / rms;

        for (size_t j = 0; j < dim; ++j) {
            float val, w;
            val = llaisys::utils::cast<float>(in_row[j]);
            w = llaisys::utils::cast<float>(weight[j]);
            
            float res = (val * inv_rms) * w;
            out_row[j] = llaisys::utils::cast<T>(res);
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, 
              llaisysDataType_t type, size_t rows, size_t dim, float eps) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out), 
                         reinterpret_cast<const float *>(in),
                         reinterpret_cast<const float *>(weight), 
                         rows, dim, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out), 
                         reinterpret_cast<const llaisys::bf16_t *>(in),
                         reinterpret_cast<const llaisys::bf16_t *>(weight), 
                         rows, dim, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out), 
                         reinterpret_cast<const llaisys::fp16_t *>(in),
                         reinterpret_cast<const llaisys::fp16_t *>(weight), 
                         rows, dim, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu