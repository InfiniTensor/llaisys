#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/sample_cpu.hpp"
#include "nvidia/sample_nvidia.cuh"

namespace llaisys::ops {
int64_t sample(tensor_t logits, int top_k, float top_p, float temperature) {
    ASSERT(logits->isContiguous(), "Sample: logits tensor must be contiguous.");
    ASSERT(logits->numel() > 0, "Sample: logits tensor must not be empty.");

    if (logits->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::sample(logits->data(), logits->dtype(), logits->numel(), top_k, top_p, temperature);
    }

    llaisys::core::context().setDevice(logits->deviceType(), logits->deviceId());
    switch (logits->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::sample(logits->data(), logits->dtype(), logits->numel(), top_k, top_p, temperature);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::sample(logits->data(), logits->dtype(), logits->numel(), top_k, top_p, temperature);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
