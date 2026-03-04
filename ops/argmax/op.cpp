#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/argmax_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/argmax_nvidia.hpp"
#endif

namespace llaisys::ops {

// 执行 argmax 操作：在输入张量中查找最大值及其索引，并写入输出张量
void argmax(tensor_t indices_out, tensor_t values_out, tensor_t input) {
    // 确保所有张量位于同一设备
    CHECK_SAME_DEVICE(indices_out, values_out, input);
    // 输入与最大值输出必须具有相同的数据类型
    CHECK_SAME_DTYPE(values_out->dtype(), input->dtype());
    // 索引输出必须为 int64 类型
    ASSERT(indices_out->dtype() == LLAISYS_DTYPE_I64, "Argmax: index output tensor must be of type int64.");
    // 输入不能为空
    ASSERT(input->numel() > 0, "Argmax: input tensor must contain at least one element.");
    // 输出张量必须为标量（单元素）
    ASSERT(indices_out->numel() == 1 && values_out->numel() == 1,
           "Argmax: both output tensors must contain exactly one element.");
    // 所有张量必须是内存连续的
    ASSERT(indices_out->isContiguous() && values_out->isContiguous() && input->isContiguous(),
           "Argmax: all input and output tensors must be contiguous in memory.");

    if (input->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(
            indices_out->data(),
            values_out->data(),
            input->data(),
            input->dtype(),
            input->numel()
        );
    }

    llaisys::core::context().setDevice(input->deviceType(), input->deviceId());

    switch (input->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::argmax(
            indices_out->data(),
            values_out->data(),
            input->data(),
            input->dtype(),
            input->numel()
        );
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:

        nvidia::argmax(
            indices_out->data(),
            values_out->data(),
            input->data(),
            input->dtype(),
            input->numel()
        );
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace llaisys::ops