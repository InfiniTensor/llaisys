#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

namespace {
// RMS 归一化内核实现：对每行输入进行均方根归一化，并应用可学习权重
template <typename T>
void rms_norm_impl(std::byte *output, const std::byte *input, const std::byte *scale,
                   size_t num_rows, size_t hidden_dim, float epsilon) {
    const T *input_data = reinterpret_cast<const T *>(input);
    const T *scale_data = reinterpret_cast<const T *>(scale);
    T *output_data = reinterpret_cast<T *>(output);

    for (size_t row = 0; row < num_rows; ++row) {
        const T *src_row = input_data + row * hidden_dim;
        T *dst_row = output_data + row * hidden_dim;

        // Step 1: 计算该行元素的平方和
        float sum_of_squares = 0.0f;
        for (size_t col = 0; col < hidden_dim; ++col) {
            float val = llaisys::utils::cast<float>(src_row[col]);
            sum_of_squares += val * val;
        }

        // Step 2: 计算均方根的倒数（RMS^-1）
        float mean_square = sum_of_squares / static_cast<float>(hidden_dim);
        float inv_rms = 1.0f / std::sqrt(mean_square + epsilon);

        // Step 3: 应用归一化和缩放
        for (size_t col = 0; col < hidden_dim; ++col) {
            float x = llaisys::utils::cast<float>(src_row[col]);
            float gamma = llaisys::utils::cast<float>(scale_data[col]);
            dst_row[col] = llaisys::utils::cast<T>(x * inv_rms * gamma);
        }
    }
}
}

namespace llaisys::ops::cpu {

// 在 CPU 上执行 RMS 归一化操作
void rms_norm(std::byte *output, const std::byte *input, const std::byte *weight,
              llaisysDataType_t dtype, size_t rows, size_t cols, float eps) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_impl<float>(output, input, weight, rows, cols, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_impl<llaisys::bf16_t>(output, input, weight, rows, cols, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_impl<llaisys::fp16_t>(output, input, weight, rows, cols, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu