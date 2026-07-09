#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    // Check device
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());

    // in and out: 2D tensors, weight: 1D tensor
    ASSERT(in->ndim() == 2, "input must be 2D");
    ASSERT(out->ndim() == 2, "output must be 2D");
    ASSERT(weight->ndim() == 1, "weight must be 1D");
    CHECK_SAME_SHAPE(out->shape(), in->shape());

    size_t rows = in->shape()[0];
    size_t cols = in->shape()[1];
    ASSERT(weight->shape()[0] == cols, "weight shape mismatch");

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(),
                            out->dtype(), rows, cols, eps);
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
