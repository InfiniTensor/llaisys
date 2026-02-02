#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/swiglu_cpu.hpp"

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_ARGUMENT(out->ndim() == 2, "out must be a 2D tensor");
    CHECK_ARGUMENT(gate->ndim() == 2, "gate must be a 2D tensor");
    CHECK_ARGUMENT(up->ndim() == 2, "up must be a 2D tensor");
    CHECK_ARGUMENT(out->shape()[0] == gate->shape()[0], "out and gate must have the same shape");
    CHECK_ARGUMENT(out->shape()[1] == gate->shape()[1], "out and gate must have the same shape");
    CHECK_ARGUMENT(out->shape()[0] == up->shape()[0], "out and up must have the same shape");
    CHECK_ARGUMENT(out->shape()[1] == up->shape()[1], "out and up must have the same shape");
    CHECK_ARGUMENT(out->isContiguous(), "swiglu: out tensor must be contiguous.");
    CHECK_ARGUMENT(gate->isContiguous(), "swiglu: gate tensor must be contiguous.");
    CHECK_ARGUMENT(up->isContiguous(), "swiglu: up tensor must be contiguous.");

    // 总是支持CPU计算
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::swiglu(out->data(), gate->data(), up->data(), out->dtype(), out->numel());
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::swiglu(out->data(), gate->data(), up->data(), out->dtype(), out->numel());
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
