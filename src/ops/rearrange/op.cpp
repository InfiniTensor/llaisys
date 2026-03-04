#include "op.hpp"

#include "../../core/llaisys_core.hpp"

#include "cpu/rearrange_cpu.hpp"

namespace llaisys::ops {

// 执行张量内存重排：将输入张量按其 strides 复制到输出张量的新内存布局中
// 要求两者形状、数据类型和设备一致，但 strides 可不同（用于实现 contiguous 或 transpose 等）
void rearrange(tensor_t output, tensor_t input) {
    // 确保输入与输出位于同一设备
    CHECK_SAME_DEVICE(output, input);
    // 数据类型必须一致
    CHECK_SAME_DTYPE(output->dtype(), input->dtype());
    // 形状必须完全相同
    ASSERT(output->shape() == input->shape(), "Rearrange: input and output shapes must be identical.");

    const size_t element_byte_size = output->elementSize();
    const auto &tensor_shape = output->shape();
    const auto &output_strides = output->strides();
    const auto &input_strides = input->strides();

    if (output->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rearrange(
            output->data(),
            input->data(),
            tensor_shape,
            output_strides,
            input_strides,
            element_byte_size
        );
    }

    llaisys::core::context().setDevice(output->deviceType(), output->deviceId());

    switch (output->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rearrange(
            output->data(),
            input->data(),
            tensor_shape,
            output_strides,
            input_strides,
            element_byte_size
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