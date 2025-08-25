#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    // Check that input tensors are on same device
    CHECK_SAME_DEVICE(out, in, weight);
    if (bias) {
        CHECK_SAME_DEVICE(out, bias);
    }
    
    // Check data types
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    if (bias) {
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
    }
    
    // Check contiguity
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), 
           "Linear: input tensors must be contiguous");
    if (bias) {
        ASSERT(bias->isContiguous(), "Linear: bias must be contiguous");
    }
    
    // Check dimensions
    ASSERT(in->ndim() == 2, "Linear: input must be 2D tensor");
    ASSERT(weight->ndim() == 2, "Linear: weight must be 2D tensor");
    ASSERT(out->ndim() == 2, "Linear: output must be 2D tensor");
    if (bias) {
        ASSERT(bias->ndim() == 1, "Linear: bias must be 1D tensor");
    }
    
    // Get dimensions
    size_t batch_size = in->shape()[0];
    size_t in_features = in->shape()[1];
    size_t out_features = weight->shape()[0];
    
    // Check shapes are compatible
    ASSERT(weight->shape()[1] == in_features, "Linear: weight input dimension must match input features");
    ASSERT(out->shape()[0] == batch_size, "Linear: output batch size must match input batch size");
    ASSERT(out->shape()[1] == out_features, "Linear: output features must match weight output features");
    if (bias) {
        ASSERT(bias->shape()[0] == out_features, "Linear: bias size must match output features");
    }

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), 
                          bias ? bias->data() : nullptr,
                          out->dtype(), batch_size, in_features, out_features);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), 
                          bias ? bias->data() : nullptr,
                          out->dtype(), batch_size, in_features, out_features);
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
