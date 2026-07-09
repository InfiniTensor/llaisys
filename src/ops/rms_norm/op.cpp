#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/rms_norm_cpu.hpp"
#include "nvidia/rms_norm_nvidia.hpp"
namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    ASSERT(out->ndim() == 2 && in->ndim() == 2 && 
    weight->ndim() == 1, "out, in must be 2D tensors, while weight must be 1D tensor");
    ASSERT(out->isContiguous() && in->isContiguous() 
    && weight->isContiguous(), "All tensors must be contiguous");
    ASSERT(out->shape() == in->shape(), "Output shape must be equal to input shape");
    ASSERT(weight->shape()[0] == in->shape()[1], "Weight shape must be equal to input shape");
    size_t batch = out->shape()[0];
    size_t in_dim = in->shape()[1];

    if (weight->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), 
        weight->data(), eps, out->dtype(), batch, in_dim);
    }

    llaisys::core::context().setDevice(weight->deviceType(), weight->deviceId());

    switch (weight->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), 
        weight->data(), eps, out->dtype(), batch, in_dim);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rms_norm(out->data(), in->data(), weight->data(), eps, out->dtype(), batch, in_dim);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
