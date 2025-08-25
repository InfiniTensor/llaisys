#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/swiglu_cpu.hpp"

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    // Check that tensors are on same device
    CHECK_SAME_DEVICE(out, gate, up);
    
    // Check data types
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());
    
    // Check contiguity
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(), 
           "SwiGLU: all tensors must be contiguous");
    
    // Check shapes are the same
    CHECK_SAME_SHAPE(out->shape(), gate->shape(), up->shape());

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::swiglu(out->data(), gate->data(), up->data(), 
                          out->dtype(), out->numel());
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::swiglu(out->data(), gate->data(), up->data(), 
                          out->dtype(), out->numel());
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
