#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/add_cpu.hpp"

namespace llaisys::ops {

// 执行两个张量的逐元素加法，结果写入输出张量
void add(tensor_t out, tensor_t lhs, tensor_t rhs) {
    // 验证三个张量位于同一计算设备
    CHECK_SAME_DEVICE(out, lhs, rhs);
    // 当前仅支持形状一致且内存连续的输入
    CHECK_SAME_SHAPE(out->shape(), lhs->shape(), rhs->shape());
    CHECK_SAME_DTYPE(out->dtype(), lhs->dtype(), rhs->dtype());
    ASSERT(out->isContiguous() && lhs->isContiguous() && rhs->isContiguous(),
           "Add operation requires all tensors to be contiguous in memory.");

    // CPU 后端始终可用
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::add(out->data(), lhs->data(), rhs->data(), out->dtype(), out->numel());
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::add(out->data(), lhs->data(), rhs->data(), out->dtype(), out->numel());
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