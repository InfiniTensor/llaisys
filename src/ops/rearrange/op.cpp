#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/rearrange_cpu.hpp"

namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in) {
    ASSERT(out->shape() == in->shape(), "Rearrange: input and output tensors must have the same shape.");
    ASSERT(out->dtype() == in->dtype(), "Rearrange: input and output tensors must have the same dtype.");

    std::vector<size_t> stride_in(in->strides().begin(), in->strides().end());
    std::vector<size_t> stride_out(out->strides().begin(), out->strides().end());

    if(out->deviceType() == LLAISYS_DEVICE_CPU) {
        return llaisys::ops::cpu::rearrange(out->data(), in->data(), out->dtype(), out->shape(), stride_in, stride_out);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rearrange(out->data(), in->data(), out->dtype(), out->shape(), stride_in, stride_out);
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
