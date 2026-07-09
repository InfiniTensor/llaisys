#include "rms_norm_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>


template <typename T>
void rms_norm_kernel(const T *input, const T *weight, T *out,
                     size_t num_rows, size_t hidden_dim, float eps) {


    for (size_t i = 0; i < num_rows; i++) {

        const T *row_in = input + i * hidden_dim;
        T *row_out = out + i * hidden_dim;

        // 计算平方和
        float sum_sq = 0.0f;
        for (size_t j = 0; j < hidden_dim; j++) {
            float val = llaisys::utils::cast<float>(row_in[j]);
            sum_sq += val * val;
        }

        // 计算 RMS 的倒数

        float mean_sq = sum_sq / static_cast<float>(hidden_dim);
        float inv_rms = 1.0f / std::sqrt(mean_sq + eps);

        // 归一化并缩放

        for (size_t j = 0; j < hidden_dim; j++) {
            float val = llaisys::utils::cast<float>(row_in[j]);
            float w = llaisys::utils::cast<float>(weight[j]);

            row_out[j] = llaisys::utils::cast<T>(val * inv_rms * w);
        }
    }
}

namespace llaisys::ops::cpu {

void rms_norm(const std::byte *input_data, const std::byte *weight_data, std::byte *out_data,
              llaisysDataType_t dtype,
              size_t num_rows, size_t hidden_dim, float eps) {

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        rms_norm_kernel<float>(
            reinterpret_cast<const float *>(input_data),
            reinterpret_cast<const float *>(weight_data),
            reinterpret_cast<float *>(out_data),
            num_rows, hidden_dim, eps);
        break;

    case LLAISYS_DTYPE_F16:
        rms_norm_kernel<llaisys::fp16_t>(
            reinterpret_cast<const llaisys::fp16_t *>(input_data),
            reinterpret_cast<const llaisys::fp16_t *>(weight_data),
            reinterpret_cast<llaisys::fp16_t *>(out_data),
            num_rows, hidden_dim, eps);
        break;

    case LLAISYS_DTYPE_BF16:
        rms_norm_kernel<llaisys::bf16_t>(
            reinterpret_cast<const llaisys::bf16_t *>(input_data),
            reinterpret_cast<const llaisys::bf16_t *>(weight_data),
            reinterpret_cast<llaisys::bf16_t *>(out_data),
            num_rows, hidden_dim, eps);
        break;

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu