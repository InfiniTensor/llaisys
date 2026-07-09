#include "op.hpp"

#include "../../utils.hpp"
#include "cpu/linear_cpu.hpp"
namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight, bias);
    // Only support contiguous tensor with same data type
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype(), bias->dtype());
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous() && bias->isContiguous(), "Linear: all tensors must be contiguous.");
    // only support 2D input and weight, 1D bias
    ASSERT(in->ndim() == 2 && weight->ndim() == 2 && bias->ndim() == 1, "Linear: in and weight must be 2D tensors, bias must be 1D tensor.");
    // out = in @ weight.T + bias
    ASSERT(in->shape()[1] == weight->shape()[1], "Linear: in.shape[1] must be equal to weight.shape[1].");
    ASSERT(out->shape()[0] == in->shape()[0] && out->shape()[1] == weight->shape()[0], "Linear: out.shape must be equal to (in.shape[0], weight.shape[0]).");
    ASSERT(bias->shape()[0] == weight->shape()[0], "Linear: bias.shape[0] must be equal to weight.shape[0].");

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), bias->data(), out->dtype(), in->shape()[0], in->shape()[1], weight->shape()[0]);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), bias->data(), out->dtype(), in->shape()[0], in->shape()[1], weight->shape()[0]);
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
