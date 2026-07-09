#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rearrange_cpu.hpp"
#include "nvidia/rearrange_nvidia.hpp"

namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in) {
    CHECK_SAME_DEVICE(out, in);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_SAME_SHAPE(out->shape(), in->shape());

    size_t numel = in->numel();
    size_t ndim = out->ndim();
    const auto &shape = out->shape();
    const auto &out_strides = out->strides();
    const auto &in_strides = in->strides();

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rearrange(out->data(), in->data(), out->dtype(), shape.data(), out_strides.data(),
                              in_strides.data(), numel, ndim);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rearrange(out->data(), in->data(), out->dtype(), shape.data(), out_strides.data(),
                              in_strides.data(), numel, ndim);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rearrange(out->data(), in->data(), out->dtype(), shape.data(), out_strides.data(),
                                 in_strides.data(), numel, ndim);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
