#include "op.hpp"
#include "../../utils.hpp"
#include "cpu/swiglu_cpu.hpp"

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    CHECK_SAME_SHAPE(out->shape(), gate->shape(), up->shape());

    ASSERT(out->ndim() == 2, "swiglu only supports 2D tensors for now");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::swiglu(out->data(), gate->data(), up->data(), out->dtype(), out->shape()[0], out->shape()[1]);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::swiglu(out->data(), gate->data(), up->data(), out->dtype(), out->shape()[0], out->shape()[1]);
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
