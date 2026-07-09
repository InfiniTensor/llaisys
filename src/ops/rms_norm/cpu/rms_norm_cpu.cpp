#include "rms_norm_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, size_t rows, size_t cols, float eps) {
    // For each row: Y_i = (W_i * X_i) / sqrt(mean(X^2) + eps)
    for (size_t r = 0; r < rows; r++) {
        const T *in_row = in + r * cols;
        T *out_row = out + r * cols;

        // Calculate RMS: sqrt(mean(x^2) + eps)
        float sum_sq = 0.0f;
        for (size_t c = 0; c < cols; c++) {
            float val;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                val = llaisys::utils::cast<float>(in_row[c]);
            } else {
                val = static_cast<float>(in_row[c]);
            }
            sum_sq += val * val;
        }

        float rms = std::sqrt(sum_sq / cols + eps);

        // Normalize and apply weight
        for (size_t c = 0; c < cols; c++) {
            float x, w;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                x = llaisys::utils::cast<float>(in_row[c]);
                w = llaisys::utils::cast<float>(weight[c]);
            } else {
                x = static_cast<float>(in_row[c]);
                w = static_cast<float>(weight[c]);
            }

            float result = (w * x) / rms;

            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                out_row[c] = llaisys::utils::cast<T>(result);
            } else {
                out_row[c] = static_cast<T>(result);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t dtype, size_t rows, size_t cols, float eps) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_<float>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight),
            rows, cols, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_<llaisys::bf16_t>(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            reinterpret_cast<const llaisys::bf16_t *>(weight),
            rows, cols, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_<llaisys::fp16_t>(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            reinterpret_cast<const llaisys::fp16_t *>(weight),
            rows, cols, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
