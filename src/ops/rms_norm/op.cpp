#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/rms_norm_nvidia.hpp"
#endif

namespace llaisys::ops {

// 执行 RMS 归一化操作：对输入的每个样本（行）进行归一化，并通过可学习缩放参数调整
void rms_norm(tensor_t output, tensor_t input, tensor_t scale, float epsilon) {
    // 确保输出、输入和缩放参数位于同一设备
    CHECK_SAME_DEVICE(output, input, scale);
    // 三者必须具有相同的数据类型
    CHECK_SAME_DTYPE(output->dtype(), input->dtype(), scale->dtype());

    // 张量维度约束
    ASSERT(output->ndim() == 2, "RMSNorm: output tensor must be 2-dimensional.");
    ASSERT(input->ndim() == 2, "RMSNorm: input tensor must be 2-dimensional.");
    ASSERT(scale->ndim() == 1, "RMSNorm: scale (weight) tensor must be 1-dimensional.");

    size_t batch_size = input->shape()[0];
    size_t hidden_dim = input->shape()[1];

    // 验证输出形状与输入一致
    ASSERT(output->shape()[0] == batch_size && output->shape()[1] == hidden_dim,
           "RMSNorm: output shape does not match input shape.");
    // 验证缩放向量长度与特征维度一致
    ASSERT(scale->shape()[0] == hidden_dim,
           "RMSNorm: length of scale vector must equal the last dimension of input.");

    // 所有张量必须内存连续
    ASSERT(output->isContiguous() && input->isContiguous() && scale->isContiguous(),
           "RMSNorm: all tensors must be contiguous in memory.");

    if (output->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(
            output->data(),
            input->data(),
            scale->data(),
            output->dtype(),
            batch_size,
            hidden_dim,
            epsilon
        );
    }

    llaisys::core::context().setDevice(output->deviceType(), output->deviceId());

    switch (output->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(
            output->data(),
            input->data(),
            scale->data(),
            output->dtype(),
            batch_size,
            hidden_dim,
            epsilon
        );
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        nvidia::rms_norm(
            output->data(),
            input->data(),
            scale->data(),
            output->dtype(),
            batch_size,
            hidden_dim,
            epsilon
        );
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace llaisys::ops