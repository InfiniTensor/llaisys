#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rearrange_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/rearrange_nvidia.cuh"
#endif

namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in) {
    CHECK_SAME_DEVICE(out, in);
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rearrange(out->data(), in->data(),
                              out->shape(), out->strides(), in->strides(),
                              out->dtype());
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rearrange(out->data(), in->data(),
                              out->shape(), out->strides(), in->strides(),
                              out->dtype());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        if (out->isContiguous() && in->isContiguous()) {
            return llaisys::core::context().runtime().api()->memcpy_async(
                out->data(),
                in->data(),
                out->numel() * out->elementSize(),
                LLAISYS_MEMCPY_D2D,
                llaisys::core::context().runtime().stream());
        }
        return nvidia::rearrange(out->data(), in->data(), out->shape(), out->strides(), in->strides(), out->dtype(),
                                 llaisys::core::context().runtime().stream());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
