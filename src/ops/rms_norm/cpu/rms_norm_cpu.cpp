#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

// RMS Normalization: Y_i = (W_i × X_i) / sqrt((1/d) * sum(X_j^2) + epsilon)
// 对每一行进行归一化
template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, 
               size_t batch_size, size_t feature_dim, float eps) {
    
    for (size_t b = 0; b < batch_size; b++) {
        // 计算当前行的指针
        const T *row_in = in + b * feature_dim;
        T *row_out = out + b * feature_dim;
        
        // 步骤1：计算平方和 (在float精度下进行)
        float sum_of_squares = 0.0f;
        for (size_t i = 0; i < feature_dim; i++) {
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                float val = llaisys::utils::cast<float>(row_in[i]);
                sum_of_squares += val * val;
            } else {
                float val = static_cast<float>(row_in[i]);
                sum_of_squares += val * val;
            }
        }
        
        // 步骤2：计算RMS (Root Mean Square)
        // rms = sqrt((1/d) * sum(x^2) + eps)
        float mean_square = sum_of_squares / static_cast<float>(feature_dim);
        float rms = std::sqrt(mean_square + eps);
        float inv_rms = 1.0f / rms;  // 归一化因子
        
        // 步骤3：应用归一化和权重: Y_i = (W_i * X_i) / rms
        for (size_t i = 0; i < feature_dim; i++) {
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                float x_val = llaisys::utils::cast<float>(row_in[i]);
                float w_val = llaisys::utils::cast<float>(weight[i]);
                float result = (w_val * x_val) * inv_rms;
                row_out[i] = llaisys::utils::cast<T>(result);
            } else {
                float x_val = static_cast<float>(row_in[i]);
                float w_val = static_cast<float>(weight[i]);
                float result = (w_val * x_val) * inv_rms;
                row_out[i] = static_cast<T>(result);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, 
              llaisysDataType_t type, size_t batch_size, size_t feature_dim, float eps) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight),
            batch_size, feature_dim, eps
        );
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            reinterpret_cast<const llaisys::bf16_t *>(weight),
            batch_size, feature_dim, eps
        );
    case LLAISYS_DTYPE_F16:
        return rms_norm_(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            reinterpret_cast<const llaisys::fp16_t *>(weight),
            batch_size, feature_dim, eps
        );
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
