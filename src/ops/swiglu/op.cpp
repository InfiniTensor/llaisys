#include "op.hpp"
#include "cpu/swiglu_cpu.hpp"
#include "llaisys.h"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/swiglu_cu.cuh"
#endif

namespace llaisys::ops {

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());
    ASSERT(out->shape() == gate->shape() && out->shape() == up->shape(),
           "swiglu(): all tensors must have the same shape");

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        cpu::swiglu(out->data(), gate->data(), up->data(), out->numel(), out->dtype());
#ifdef ENABLE_NVIDIA_API
    } else if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        nvidia::swiglu(out->data(), gate->data(), up->data(), out->numel(), out->dtype());
#endif
    } else
        EXCEPTION_UNSUPPORTED_DEVICE;
}

} // namespace llaisys::ops
