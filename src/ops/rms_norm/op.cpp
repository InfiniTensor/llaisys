#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    ASSERT(in->shape().size() == 2, "RMSNorm: input tensor must be 2-D.");
    ASSERT(weight->shape().size() == 1, "RMSNorm: weight tensor must be 1-D.");
    ASSERT(out->shape().size() == 2, "RMSNorm: output tensor must be 2-D.");
    size_t dimi = in->shape()[0];
    size_t dimj = in->shape()[1];
    
    ASSERT(weight->shape()[0] == dimj, "RMSNorm: weight tensor shape is invalid.");
    ASSERT(out->shape()[0] == dimi && out->shape()[1] == dimj, "RMSNorm: output tensor shape is invalid.");

    if(out->deviceType() == LLAISYS_DEVICE_CPU) {
        return llaisys::ops::cpu::rms_norm(out->data(), in->data(), weight->data(), eps, out->dtype(), {dimi, dimj});
    }
    
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(), eps, out->dtype(), {dimi, dimj});
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
