#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <stdlib.h>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, const float eps,
               const size_t B, const size_t K) {
    for (size_t i = 0; i < B; ++i) {
        float sum_sq = 0.0f;

        // x 按行求平方和
        for (size_t j = 0; j < K; ++j) {
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                float val = llaisys::utils::cast<float>(in[i * K + j]);
                sum_sq += val * val;
            } else {
                float val = in[i * K + j];
                sum_sq += val * val;
            }
        }

        // 按行求平方根
        float scale = 1.0f / std::sqrt(sum_sq / K + eps);

        // x 和 w 相同位置元素相乘 除以 平方根
        for (size_t j = 0; j < K; ++j) {
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                out[i * K + j] = llaisys::utils::cast<T>(llaisys::utils::cast<float>(in[i * K + j]) * llaisys::utils::cast<float>(weight[j]) * scale);
            } else {
                out[i * K + j] = in[i * K + j] * weight[j] * scale;
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, const float eps,
              llaisysDataType_t type, const size_t B, const size_t K) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(weight),
                         eps, B, K);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in), reinterpret_cast<const llaisys::bf16_t *>(weight),
                         eps, B, K);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in), reinterpret_cast<const llaisys::fp16_t *>(weight),
                         eps, B, K);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
