#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/sample_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "cuda/sample_cuda.cuh"
#endif

namespace llaisys::ops {
void sample(tensor_t out_idx, tensor_t logits, float temperature, int top_k, float top_p) {
    ASSERT(logits->isContiguous(), "Sample: logits must be contiguous.");

    if (logits->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::sample(out_idx->data(), logits->data(), logits->dtype(), logits->numel(),
                           temperature, top_k, top_p);
    }

    llaisys::core::context().setDevice(logits->deviceType(), logits->deviceId());

    switch (logits->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::sample(out_idx->data(), logits->data(), logits->dtype(), logits->numel(),
                           temperature, top_k, top_p);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return cuda::sample(out_idx->data(), logits->data(), logits->dtype(), logits->numel(),
                            temperature, top_k, top_p);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
