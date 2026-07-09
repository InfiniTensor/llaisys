#include "rms_norm_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, size_t batch, size_t dim, float eps) {
    for (size_t b = 0; b < batch; b++) {
        float sum_sq = 0.0f;
        for (size_t i = 0; i < dim; i++) {
            float val = llaisys::utils::cast<float>(in[b * dim + i]);
            sum_sq += val * val;
        }
        float rms = std::sqrt(sum_sq / dim + eps);
        for (size_t i = 0; i < dim; i++) {
            float val = llaisys::utils::cast<float>(in[b * dim + i]);
            float w = llaisys::utils::cast<float>(weight[i]);
            out[b * dim + i] = llaisys::utils::cast<T>(w * val / rms);
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, llaisysDataType_t type,
              size_t batch, size_t dim, float eps) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                        reinterpret_cast<const float *>(weight), batch, dim, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                        reinterpret_cast<const llaisys::bf16_t *>(weight), batch, dim, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                        reinterpret_cast<const llaisys::fp16_t *>(weight), batch, dim, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
}
