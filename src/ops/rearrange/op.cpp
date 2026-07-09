#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/rearrange_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/rearrange_nvidia.hpp"
#endif

namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in) {
    CHECK_SAME_DEVICE(out, in);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    ASSERT(out->shape() == in->shape(), "Rearrange: shape mismatch.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        // 直接传递，类型现在匹配 std::vector<ptrdiff_t>
        return cpu::rearrange(out->data(), in->data(), out->dtype(),
                              in->shape(), in->strides(), out->strides());
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rearrange(out->data(), in->data(), out->dtype(), in->shape(), in->strides(), out->strides());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rearrange(out->data(), in->data(), out->dtype(), in->shape(), in->strides(), out->strides());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
}
