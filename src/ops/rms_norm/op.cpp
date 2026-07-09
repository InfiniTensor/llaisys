#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "cuda/rms_norm_cuda.cuh"
#endif


namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);

    
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());

    
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), 
           "RMS Norm: all tensors must be contiguous");

    
    ASSERT(in->ndim() == 2, "RMS Norm: input must be 2D tensor");
    ASSERT(weight->ndim() == 1, "RMS Norm: weight must be 1D tensor");
    ASSERT(out->ndim() == 2, "RMS Norm: output must be 2D tensor");

    
    size_t batch_size = in->shape()[0];
    size_t feature_dim = in->shape()[1];
    ASSERT(out->shape()[0] == batch_size, "RMS Norm: output batch size must match input");
    ASSERT(out->shape()[1] == feature_dim, "RMS Norm: output feature dim must match input");
    ASSERT(weight->shape()[0] == feature_dim, "RMS Norm: weight size must match feature dim");

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
    return cuda::rms_norm(out->data(), in->data(), weight->data(), out->dtype(), batch_size, feature_dim, eps);
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
