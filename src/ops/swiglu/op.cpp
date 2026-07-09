#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/swiglu_cpu.hpp"
#include "nvidia/swiglu_nvidia.hpp"

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());
    CHECK_SAME_SHAPE(out->shape(), gate->shape(), up->shape());
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(),
           "swiglu: all tensors must be contiguous.");
    size_t seq_len = out->shape()[0];
    size_t inter_size = out->shape()[1];

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::swiglu(out->data(), gate->data(), up->data(), 
        out->dtype(), seq_len, inter_size);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::swiglu(out->data(), gate->data(), up->data(), 
        out->dtype(), seq_len, inter_size);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::swiglu(out->data(), gate->data(), up->data(), out->dtype(), seq_len, inter_size);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
