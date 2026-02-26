#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/rms_norm_nvidia.cuh"
#endif

#ifdef ENABLE_ILUVATAR_API
#include "iluvatar/rms_norm_iluvatar.cuh"
#endif

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    
    // Only support 2D tensors for now
    ASSERT(in->ndim() == 2, "RMSNorm: in must be 2D tensor.");
    ASSERT(out->ndim() == 2, "RMSNorm: out must be 2D tensor.");
    ASSERT(weight->ndim() == 1, "RMSNorm: weight must be 1D tensor.");
    ASSERT(in->dtype() == out->dtype() && in->dtype() == weight->dtype(), "RMSNorm: in, out, weight must have same dtype.");
    ASSERT(in->shape() == out->shape(), "RMSNorm: in and out must have same shape.");
    ASSERT(weight->shape()[0] == in->shape()[1], "RMSNorm: weight shape[0] must match in shape[1].");
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "RMSNorm: all tensors must be contiguous.");

    size_t batch_size = in->shape()[0];
    size_t hidden_dim = in->shape()[1];

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(
            out->data(), 
            in->data(), 
            weight->data(), 
            in->dtype(), 
            eps, 
            batch_size, 
            hidden_dim
        );
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(
            out->data(), 
            in->data(), 
            weight->data(), 
            in->dtype(), 
            eps, 
            batch_size, 
            hidden_dim
        );
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rms_norm(
            out->data(), 
            in->data(), 
            weight->data(), 
            in->dtype(), 
            eps, 
            batch_size, 
            hidden_dim
        );
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops