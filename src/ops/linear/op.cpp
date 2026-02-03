#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {

// 执行线性变换：output = input @ weight^T + bias（若提供）
void linear(tensor_t output, tensor_t input, tensor_t weight_matrix, tensor_t bias_vector) {
    // 确保输出、输入和权重位于同一设备
    CHECK_SAME_DEVICE(output, input, weight_matrix);
    if (bias_vector) {
        // 若存在偏置，也需在同一设备且数据类型一致
        CHECK_SAME_DEVICE(output, bias_vector);
        CHECK_SAME_DTYPE(output->dtype(), bias_vector->dtype());
    }
    // 输出、输入和权重必须具有相同的数据类型
    CHECK_SAME_DTYPE(output->dtype(), input->dtype(), weight_matrix->dtype());

    // 所有张量维度要求：均为二维（偏置除外）
    ASSERT(output->ndim() == 2, "Linear: output tensor must be 2-dimensional.");
    ASSERT(input->ndim() == 2, "Linear: input tensor must be 2-dimensional.");
    ASSERT(weight_matrix->ndim() == 2, "Linear: weight matrix must be 2-dimensional.");

    size_t batch_size = input->shape()[0];
    size_t in_features = input->shape()[1];
    size_t out_features = weight_matrix->shape()[0]; // 权重形状为 [out_features, in_features]

    // 验证权重的输入特征维度与输入匹配
    ASSERT(weight_matrix->shape()[1] == in_features,
           "Linear: mismatch between input feature dimension and weight's second dimension.");
    // 验证输出形状是否符合预期
    ASSERT(output->shape()[0] == batch_size && output->shape()[1] == out_features,
           "Linear: output tensor shape does not match expected (batch_size, out_features).");

    // 若提供偏置，必须为一维且长度等于输出特征数
    if (bias_vector) {
        ASSERT(bias_vector->ndim() == 1 && bias_vector->shape()[0] == out_features,
               "Linear: bias must be a 1D tensor with length equal to out_features.");
    }

    // 所有参与计算的张量必须内存连续
    ASSERT(output->isContiguous() && input->isContiguous() && weight_matrix->isContiguous()
               && (!bias_vector || bias_vector->isContiguous()),
           "Linear: all input and output tensors must be contiguous in memory.");

    if (output->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(
            output->data(),
            input->data(),
            weight_matrix->data(),
            bias_vector ? bias_vector->data() : nullptr,
            output->dtype(),
            batch_size,
            out_features,
            in_features
        );
    }

    llaisys::core::context().setDevice(output->deviceType(), output->deviceId());

    switch (output->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(
            output->data(),
            input->data(),
            weight_matrix->data(),
            bias_vector ? bias_vector->data() : nullptr,
            output->dtype(),
            batch_size,
            out_features,
            in_features
        );
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace llaisys::ops