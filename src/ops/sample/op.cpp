#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/sample_cpu.hpp"

namespace llaisys::ops {
void sample(tensor_t sampled_token, tensor_t logits, float temperature, int top_k, float top_p) {
    CHECK_SAME_DEVICE(sampled_token, logits);
    ASSERT(logits->ndim() == 1, "sample: logits must be 1D.");
    ASSERT(sampled_token->numel() == 1, "sample: sampled_token must have one element.");
    ASSERT(sampled_token->dtype() == LLAISYS_DTYPE_I64, "sample: sampled_token must be int64.");
    ASSERT(sampled_token->isContiguous() && logits->isContiguous(), "sample: all tensors must be contiguous.");
    ASSERT(temperature > 0.0f, "sample: temperature must be positive.");
    ASSERT(top_k >= 0, "sample: top_k must be non-negative.");
    ASSERT(top_p >= 0.0f && top_p <= 1.0f, "sample: top_p must be in [0, 1].");

    if (logits->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::sample(sampled_token->data(), logits->data(), logits->dtype(), logits->numel(), temperature, top_k, top_p);
    }

    llaisys::core::context().setDevice(logits->deviceType(), logits->deviceId());

    switch (logits->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::sample(sampled_token->data(), logits->data(), logits->dtype(), logits->numel(), temperature, top_k, top_p);
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
