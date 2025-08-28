#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/swiglu_cpu.hpp"

namespace llaisys::ops {

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    // 
    CHECK_SAME_DEVICE(out, gate, up);

    // 
    ASSERT(out->ndim() == 2 && gate->ndim() == 2 && up->ndim() == 2,
           "swiglu: all tensors must be 2D [N, D]");
    const size_t N = out->shape()[0];
    const size_t D = out->shape()[1];
    ASSERT(gate->shape()[0] == N && gate->shape()[1] == D, "swiglu: gate shape mismatch");
    ASSERT(up->shape()[0] == N && up->shape()[1] == D, "swiglu: up shape mismatch");

    // dtype
    ASSERT(out->dtype() == gate->dtype() && out->dtype() == up->dtype(),
           "swiglu: dtypes of out/gate/up must match");

    //
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(),
           "swiglu: all tensors must be contiguous");

    // 
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::swiglu(out->data(), gate->data(), up->data(),
                           out->dtype(), N, D);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::swiglu(out->data(), gate->data(), up->data(),
                           out->dtype(), N, D);
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
