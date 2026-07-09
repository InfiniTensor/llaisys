#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "linear: all tensors must be contiguous.");

    ASSERT(out->ndim() == 2 && in->ndim() == 2 && weight->ndim() == 2, "linear: out/in/weight must be 2D");
    const size_t B = in->shape()[0];
    const size_t K = in->shape()[1];
    const size_t M = weight->shape()[0];
    ASSERT(weight->shape()[1] == K, "linear: weight shape mismatch (expect [M, K])");
    ASSERT(out->shape()[0] == B && out->shape()[1] == M, "linear: out shape mismatch (expect [B, M])");

    auto bias_ptr = bias ? bias->data() : nullptr;
    if (bias_ptr) {
        CHECK_SAME_DEVICE(out, bias);
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
        ASSERT(bias->isContiguous(), "linear: all tensors must be contiguous.");
        ASSERT(bias->ndim() == 1 && bias->shape()[0] == M, "linear: bias shape mismatch (expect [M])");
    }
    // always support cpu calculation
    if (weight->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), bias_ptr, weight->dtype(), B, K, M);
    }

    llaisys::core::context().setDevice(weight->deviceType(), weight->deviceId());

    switch (weight->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), bias_ptr, weight->dtype(), B, K, M);
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
