#include "op.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    // Only support contiguous inputs with same shape for now.
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "RMSNorm: all tensors must be contiguous.");
    // Only support 2D input and output for now.
    ASSERT(in->ndim() == 2, "RMSNorm: only support 2D input tensor.");
    // Check weight shape
    ASSERT(weight->ndim() == 1 && weight->shape()[0] == in->shape()[1], "RMSNorm: weight shape is invalid.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(), out->dtype(), in->shape()[0], in->shape()[1], eps);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(), out->dtype(), in->shape()[0], in->shape()[1], eps);
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
