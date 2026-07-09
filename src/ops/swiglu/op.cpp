#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/swiglu_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/swiglu_cuda.cuh"
#endif

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    // 检查设备一致性
    CHECK_SAME_DEVICE(out, gate, up);

    // 检查数据类型一致性
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());

    // 检查形状一致性
    CHECK_SAME_SHAPE(out->shape(), gate->shape(), up->shape());

    // 检查所有张量都是连续的
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(),
           "swiglu: all tensors must be contiguous.");

    // 设置设备上下文
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    size_t numel = out->numel();

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::swiglu(out->data(), gate->data(), up->data(), out->dtype(), numel);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::swiglu(out->data(), gate->data(), up->data(), out->dtype(), numel);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
