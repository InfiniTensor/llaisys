#include "rms_norm_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, size_t input_rows, size_t input_cols, float eps) {
    for (size_t i = 0; i < input_rows; i++) {
        float sum_squares = 0.0f;
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            for (size_t j = 0; j < input_cols; j++) {
                float val = llaisys::utils::cast<float>(in[i * input_cols + j]);
                sum_squares += val * val;
            }

            float rms = std::sqrt(sum_squares / input_cols + eps);
            for (size_t j = 0; j < input_cols; j++) {
                float val = llaisys::utils::cast<float>(in[i * input_cols + j]);
                out[i * input_cols + j] = llaisys::utils::cast<T>(val * llaisys::utils::cast<float>(weight[j]) / rms);
            }
        } else {
            for (size_t j = 0; j < input_cols; j++) {
                float val = static_cast<float>(in[i * input_cols + j]);
                sum_squares += val * val;
            }

            float rms = std::sqrt(sum_squares / input_cols + eps);
            for (size_t j = 0; j < input_cols; j++) {
                float val = static_cast<float>(in[i * input_cols + j]);
                out[i * input_cols + j] = static_cast<T>(val * static_cast<float>(weight[j]) / rms);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, llaisysDataType_t type, size_t input_rows, size_t input_cols, float eps) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(weight), input_rows, input_cols, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                         reinterpret_cast<const llaisys::bf16_t *>(weight), input_rows, input_cols, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                         reinterpret_cast<const llaisys::fp16_t *>(weight), input_rows, input_cols, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu