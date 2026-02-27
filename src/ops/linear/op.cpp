#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/linear_nvidia.cuh"
#endif

#ifdef ENABLE_METAX_API
#include "metax/linear_metax.cuh"
#endif

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    if (bias) {
        CHECK_SAME_DEVICE(out, bias);
    }
    
    // Only support 2D tensors for now
    ASSERT(in->ndim() == 2, "Linear: in must be 2D tensor.");
    ASSERT(weight->ndim() == 2, "Linear: weight must be 2D tensor.");
    ASSERT(out->ndim() == 2, "Linear: out must be 2D tensor.");
    ASSERT(in->dtype() == out->dtype() && in->dtype() == weight->dtype(), "Linear: in, out, weight must have same dtype.");
    ASSERT(in->shape()[1] == weight->shape()[1], "Linear: in shape[1] must match weight shape[1].");
    ASSERT(out->shape()[0] == in->shape()[0], "Linear: out shape[0] must match in shape[0].");
    ASSERT(out->shape()[1] == weight->shape()[0], "Linear: out shape[1] must match weight shape[0].");
    
    if (bias) {
        ASSERT(bias->ndim() == 1, "Linear: bias must be 1D tensor.");
        ASSERT(bias->shape()[0] == out->shape()[1], "Linear: bias shape[0] must match out shape[1].");
        ASSERT(bias->dtype() == in->dtype(), "Linear: bias must have same dtype as in.");
        ASSERT(bias->isContiguous(), "Linear: bias must be contiguous.");
    }
    
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "Linear: all tensors must be contiguous.");

    size_t batch_size = in->shape()[0];
    size_t in_dim = in->shape()[1];
    size_t out_dim = out->shape()[1];

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(
            out->data(), 
            in->data(), 
            weight->data(), 
            bias ? bias->data() : nullptr, 
            in->dtype(), 
            batch_size, 
            in_dim, 
            out_dim
        );
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(
            out->data(), 
            in->data(), 
            weight->data(), 
            bias ? bias->data() : nullptr, 
            in->dtype(), 
            batch_size, 
            in_dim, 
            out_dim
        );
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::linear(
            out->data(), 
            in->data(), 
            weight->data(), 
            bias ? bias->data() : nullptr, 
            in->dtype(), 
            batch_size, 
            in_dim, 
            out_dim
        );
#endif
#ifdef ENABLE_METAX_API
    case LLAISYS_DEVICE_METAX:
        return metax::linear(
            out->data(), 
            in->data(), 
            weight->data(), 
            bias ? bias->data() : nullptr, 
            in->dtype(), 
            batch_size, 
            in_dim, 
            out_dim
        );
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops