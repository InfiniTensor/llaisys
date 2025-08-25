#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    // Check that tensors are on same device
    CHECK_SAME_DEVICE(out, in, weight);
    
    // Check data types
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    
    // Check contiguity
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), 
           "RMS Norm: all tensors must be contiguous");
    
    // Check dimensions
    ASSERT(in->ndim() == 2, "RMS Norm: input must be 2D tensor");
    ASSERT(weight->ndim() == 1, "RMS Norm: weight must be 1D tensor");
    ASSERT(out->ndim() == 2, "RMS Norm: output must be 2D tensor");
    
    // Check shapes are compatible
    size_t batch_size = in->shape()[0];
    size_t feature_dim = in->shape()[1];
    ASSERT(out->shape()[0] == batch_size, "RMS Norm: output batch size must match input");
    ASSERT(out->shape()[1] == feature_dim, "RMS Norm: output feature dim must match input");
    ASSERT(weight->shape()[0] == feature_dim, "RMS Norm: weight size must match feature dim");

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(), 
                            out->dtype(), batch_size, feature_dim, eps);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(), 
                            out->dtype(), batch_size, feature_dim, eps);
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
