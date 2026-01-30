#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/add_cpu.hpp"

namespace llaisys::ops {
// 接收三个 tensor_t（即 std::shared_ptr<Tensor>）类型的参数。c 是输出张量，a 和 b 是输入张量   
void add(tensor_t c, tensor_t a, tensor_t b) {
    // 设备一致性：
    CHECK_SAME_DEVICE(c, a, b);
    // Only support contiguous inputs with same shape for now.
    // 形状一致性
    CHECK_SAME_SHAPE(c->shape(), a->shape(), b->shape());
    // 类型一致性
    CHECK_SAME_DTYPE(c->dtype(), a->dtype(), b->dtype());
    // 内存连续性
    ASSERT(c->isContiguous() && a->isContiguous() && b->isContiguous(), "Add: all tensors must be contiguous.");

    // always support cpu calculation
    // 如果检测到设备是 CPU，直接调用 cpu::add 函数（定义在 cpu/add_cpu.hpp 中）
    if (c->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::add(c->data(), a->data(), b->data(), c->dtype(), c->numel());
    }
    // 在执行具体的计算或内存操作前，必须告诉底层运行时系统（Runtime）当前要在哪个硬件设备上工作。
    llaisys::core::context().setDevice(c->deviceType(), c->deviceId());

    switch (c->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::add(c->data(), a->data(), b->data(), c->dtype(), c->numel());
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
