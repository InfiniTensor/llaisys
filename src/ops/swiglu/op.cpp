#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/swiglu_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/swiglu_nvidia.cuh"
#endif

#ifdef ENABLE_METAX_API
#include "metax/swiglu_metax.cuh"
#endif

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    
    // Only support 2D tensors for now
    ASSERT(gate->ndim() == 2, "SwiGLU: gate must be 2D tensor.");
    ASSERT(up->ndim() == 2, "SwiGLU: up must be 2D tensor.");
    ASSERT(out->ndim() == 2, "SwiGLU: out must be 2D tensor.");
    ASSERT(gate->dtype() == up->dtype() && gate->dtype() == out->dtype(), "SwiGLU: all tensors must have same dtype.");
    ASSERT(gate->shape() == up->shape() && gate->shape() == out->shape(), "SwiGLU: all tensors must have same shape.");
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(), "SwiGLU: all tensors must be contiguous.");

    size_t seq_len = gate->shape()[0];
    size_t intermediate_size = gate->shape()[1];

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::swiglu(
            out->data(), 
            gate->data(), 
            up->data(), 
            gate->dtype(), 
            seq_len, 
            intermediate_size
        );
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::swiglu(
            out->data(), 
            gate->data(), 
            up->data(), 
            gate->dtype(), 
            seq_len, 
            intermediate_size
        );
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::swiglu(
            out->data(), 
            gate->data(), 
            up->data(), 
            gate->dtype(), 
            seq_len, 
            intermediate_size
        );
#endif
#ifdef ENABLE_METAX_API
    case LLAISYS_DEVICE_METAX:
        return metax::swiglu(
            out->data(), 
            gate->data(), 
            up->data(), 
            gate->dtype(), 
            seq_len * intermediate_size
        );
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops