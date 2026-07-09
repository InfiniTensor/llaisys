#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/rms_norm_cpu.hpp"

namespace llaisys::ops {

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {

    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());

    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "RMSNorm: tensors must be contiguous.");

    ASSERT(in->ndim() == 2, "RMSNorm: input must be 2D.");
    ASSERT(out->ndim() == 2, "RMSNorm: output must be 2D.");
    ASSERT(weight->ndim() == 1, "RMSNorm: weight must be 1D.");

    ASSERT(in->shape()[0] == out->shape()[0], "RMSNorm: input/output rows mismatch.");
    ASSERT(in->shape()[1] == out->shape()[1], "RMSNorm: input/output cols mismatch.");

    ASSERT(weight->shape()[0] == in->shape()[1],
           "RMSNorm: weight size must match input hidden dim.");

    size_t num_rows = in->shape()[0];
    size_t hidden_dim = in->shape()[1];

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        ::llaisys::ops::cpu::rms_norm(
            in->data(),
            weight->data(),
            out->data(),
            in->dtype(),
            num_rows,
            hidden_dim,
            eps);
        return;
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        ::llaisys::ops::cpu::rms_norm(
            in->data(), weight->data(), out->data(),
            in->dtype(), num_rows, hidden_dim, eps);
        return;

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
