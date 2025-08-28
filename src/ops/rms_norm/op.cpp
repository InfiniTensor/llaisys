#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"

namespace llaisys::ops {

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    // --- device checks ---
    CHECK_SAME_DEVICE(out, in, weight);

    // --- shape checks ---
    ASSERT(out->ndim() == 2, "rms_norm: out must be 2D");
    ASSERT(in->ndim() == 2, "rms_norm: in must be 2D");
    ASSERT(weight->ndim() == 1, "rms_norm: weight must be 1D");

    const size_t N = in->shape()[0];
    const size_t d = in->shape()[1];

    ASSERT(out->shape()[0] == N && out->shape()[1] == d,
           "rms_norm: out shape must match in [N,d]");
    ASSERT(weight->shape()[0] == d,
           "rms_norm: weight length must equal d");

    // --- dtype checks ---
    ASSERT(out->dtype() == in->dtype() && out->dtype() == weight->dtype(),
           "rms_norm: out/in/weight dtypes must match");
    // 

    // --- contiguity ---
    ASSERT(out->isContiguous(), "rms_norm: out must be contiguous");
    ASSERT(in->isContiguous(), "rms_norm: in must be contiguous");
    ASSERT(weight->isContiguous(), "rms_norm: weight must be contiguous");

    // --- dispatch ---
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(),
                             in->data(),
                             weight->data(),
                             out->dtype(),
                             N, d, eps);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(),
                             in->data(),
                             weight->data(),
                             out->dtype(),
                             N, d, eps);
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
