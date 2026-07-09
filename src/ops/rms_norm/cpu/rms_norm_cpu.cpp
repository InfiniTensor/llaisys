#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

// RMS Normalization: Y_i = (W_i × X_i) / sqrt((1/d) * sum(X_j^2) + eps)
template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, size_t num_rows, size_t row_dim, float eps) {
    // 对每一行进行归一化
    for (size_t r = 0; r < num_rows; r++) {
        const T *in_row = in + r * row_dim;
        T *out_row = out + r * row_dim;

        // 计算均方根: sqrt((1/d) * sum(X_j^2) + eps)
        float sum_squares = 0.0f;
        for (size_t i = 0; i < row_dim; i++) {
            float val;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                val = llaisys::utils::cast<float>(in_row[i]);
            } else {
                val = static_cast<float>(in_row[i]);
            }
            sum_squares += val * val;
        }

        // 计算 RMS: 1 / sqrt((1/d) * sum + eps)
        float rms = 1.0f / std::sqrt(sum_squares / row_dim + eps);

        // 应用归一化和权重: Y_i = (W_i × X_i) × rms
        for (size_t i = 0; i < row_dim; i++) {
            float x_val, w_val;

            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                x_val = llaisys::utils::cast<float>(in_row[i]);
                w_val = llaisys::utils::cast<float>(weight[i]);
            } else {
                x_val = static_cast<float>(in_row[i]);
                w_val = static_cast<float>(weight[i]);
            }

            float result = x_val * rms * w_val;

            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                out_row[i] = llaisys::utils::cast<T>(result);
            } else {
                out_row[i] = static_cast<T>(result);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t type, size_t num_rows, size_t row_dim, float eps) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                         reinterpret_cast<const float *>(weight), num_rows, row_dim, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                         reinterpret_cast<const llaisys::bf16_t *>(weight), num_rows, row_dim, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                         reinterpret_cast<const llaisys::fp16_t *>(weight), num_rows, row_dim, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
