#include "rms_norm_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, size_t M, size_t d, float eps) {
    for (size_t i = 0; i < M; ++i) {
        float sum_sq = 0.0f;
        const T* row_in = in + i * d;
        T* row_out = out + i * d;

        // 1. 计算当前行元素的平方和
        for (size_t j = 0; j < d; ++j) {
            float val = llaisys::utils::cast<float>(row_in[j]);
            sum_sq += val * val;
        }

        // 2. 计算 RMS (均方根)
        float rms = std::sqrt(sum_sq / static_cast<float>(d) + eps);

        // 3. 归一化并应用权重 W
        for (size_t j = 0; j < d; ++j) {
            float val = llaisys::utils::cast<float>(row_in[j]);
            float w = llaisys::utils::cast<float>(weight[j]);
            row_out[j] = llaisys::utils::cast<T>(w * val / rms);
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, 
              llaisysDataType_t type, size_t M, size_t d, float eps) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                         reinterpret_cast<const float *>(weight), M, d, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                         reinterpret_cast<const llaisys::bf16_t *>(weight), M, d, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                         reinterpret_cast<const llaisys::fp16_t *>(weight), M, d, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu