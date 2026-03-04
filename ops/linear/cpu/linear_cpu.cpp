#include "linear_cpu.hpp"

#include "../../../utils.hpp"

#include <cstddef>

namespace {
    template <typename T>
    void linear_impl(std::byte *output, const std::byte *input, const std::byte *weight_matrix,
                     const std::byte *bias_vector, size_t batch_size, size_t out_features, size_t in_features) {
        const T *input_data = reinterpret_cast<const T *>(input);
        const T *weight_data = reinterpret_cast<const T *>(weight_matrix);
        const T *bias_data = bias_vector ? reinterpret_cast<const T *>(bias_vector) : nullptr;
        T *output_data = reinterpret_cast<T *>(output);

        // 遍历每个输入样本（batch 维度）
        for (size_t row = 0; row < batch_size; ++row) {
            // 遍历每个输出特征
            for (size_t col = 0; col < out_features; ++col) {
                // 初始化累加器：若有偏置，则加载对应偏置项；否则为 0
                float accumulator = bias_data ? llaisys::utils::cast<float>(bias_data[col]) : 0.0f;

                // 获取当前输出特征对应的权重行（weight shape: [out_features, in_features]）
                const T *weight_row = weight_data + col * in_features;
                // 获取当前输入样本的特征向量
                const T *input_row = input_data + row * in_features;

                // 计算输入与权重的点积
                for (size_t idx = 0; idx < in_features; ++idx) {
                    accumulator += llaisys::utils::cast<float>(input_row[idx]) *
                                   llaisys::utils::cast<float>(weight_row[idx]);
                }

                // 将结果写回输出张量
                output_data[row * out_features + col] = llaisys::utils::cast<T>(accumulator);
            }
        }
    }
}

namespace llaisys::ops::cpu {

// 在 CPU 上执行线性变换：output = input @ weight^T + bias
void linear(std::byte *output, const std::byte *input, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t dtype, size_t m, size_t n, size_t k) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return linear_impl<float>(output, input, weight, bias, m, n, k);
    case LLAISYS_DTYPE_BF16:
        return linear_impl<llaisys::bf16_t>(output, input, weight, bias, m, n, k);
    case LLAISYS_DTYPE_F16:
        return linear_impl<llaisys::fp16_t>(output, input, weight, bias, m, n, k);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu