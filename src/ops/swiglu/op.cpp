#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/swiglu_cpu.hpp"

namespace llaisys::ops {

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {

    CHECK_SAME_DEVICE(out, gate, up);
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());

    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(),
           "SwiGLU: tensors must be contiguous.");

    ASSERT(gate->ndim() == 2, "SwiGLU: gate must be 2D.");
    ASSERT(up->ndim() == 2, "SwiGLU: up must be 2D.");
    ASSERT(out->ndim() == 2, "SwiGLU: out must be 2D.");

    ASSERT(gate->shape() == up->shape(), "SwiGLU: gate and up shapes mismatch.");
    ASSERT(out->shape() == gate->shape(), "SwiGLU: out shape mismatch.");

    size_t numel = gate->numel();

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        ::llaisys::ops::cpu::swiglu(
            gate->data(),
            up->data(),
            out->data(),
            out->dtype(),
            numel);
        return;
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        ::llaisys::ops::cpu::swiglu(
            gate->data(), up->data(), out->data(),
            out->dtype(), numel);
        return;

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
