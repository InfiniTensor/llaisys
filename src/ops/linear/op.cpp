#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    
    CHECK_SAME_DEVICE(out, in, weight);
    if (bias) {
        CHECK_SAME_DEVICE(out, bias);
    }

    
    ASSERT(out->ndim() == 2, "linear: out must be 2D");
    ASSERT(in->ndim() == 2, "linear: in must be 2D");
    ASSERT(weight->ndim() == 2, "linear: weight must be 2D");

    const size_t N = in->shape()[0];
    const size_t K = in->shape()[1];
    const size_t M = weight->shape()[0];
    ASSERT(weight->shape()[1] == K, "linear: weight.shape[1] must equal in.shape[1]");
    ASSERT(out->shape()[0] == N && out->shape()[1] == M,
           "linear: out.shape must be [N, M]");

    
    ASSERT(out->dtype() == in->dtype() && out->dtype() == weight->dtype(),
           "linear: out/in/weight dtypes must match");
    if (bias) {
        ASSERT(bias->ndim() == 1, "linear: bias must be 1D");
        ASSERT(bias->shape()[0] == M, "linear: bias.size must equal M");
        ASSERT(bias->dtype() == out->dtype(),
               "linear: bias dtype must match out dtype");
    }

   
    ASSERT(out->isContiguous(), "linear: out must be contiguous");
    ASSERT(in->isContiguous(), "linear: in must be contiguous");
    ASSERT(weight->isContiguous(), "linear: weight must be contiguous");
    if (bias) {
        ASSERT(bias->isContiguous(), "linear: bias must be contiguous");
    }

    
    const std::byte *bias_ptr = bias ? bias->data() : nullptr;

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), bias_ptr,
                           out->dtype(), N, M, K);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), bias_ptr,
                           out->dtype(), N, M, K);
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
