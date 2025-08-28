#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rearrange_cpu.hpp"

namespace llaisys::ops {

void rearrange(tensor_t out, tensor_t in) {

    CHECK_SAME_DEVICE(out, in);


    ASSERT(out->ndim() == in->ndim(), "rearrange: rank mismatch");
    CHECK_SAME_SHAPE(out->shape(), in->shape());

    // dtype
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());


    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rearrange(out->data(), in->data(), out->dtype(),
                              out->shape(), out->strides(), in->strides());
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rearrange(out->data(), in->data(), out->dtype(),
                              out->shape(), out->strides(), in->strides());
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
