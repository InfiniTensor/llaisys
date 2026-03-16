#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/random_sample_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/random_sample_nvidia.hpp"
#endif

#include <vector>

namespace llaisys::ops {

void random_sample(
    tensor_t out_idx,
    tensor_t logits,
    float temperature,
    size_t top_k,
    float top_p,
    uint64_t seed) {
    CHECK_SAME_DEVICE(out_idx, logits);
    ASSERT(out_idx->dtype() == LLAISYS_DTYPE_I64, "RandomSample: out_idx must be Int64.");
    ASSERT(out_idx->numel() == 1, "RandomSample: out_idx must have one element.");
    ASSERT(logits->isContiguous(), "RandomSample: logits must be contiguous.");
    ASSERT(logits->numel() > 0, "RandomSample: logits must be non-empty.");

    if (logits->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::random_sample(
            out_idx->data(),
            logits->data(),
            logits->dtype(),
            logits->numel(),
            temperature,
            top_k,
            top_p,
            seed);
    }

    llaisys::core::context().setDevice(logits->deviceType(), logits->deviceId());

#ifdef ENABLE_NVIDIA_API
    if (logits->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        return nvidia::random_sample(
            out_idx->data(),
            logits->data(),
            logits->dtype(),
            logits->numel(),
            temperature,
            top_k,
            top_p,
            seed);
    }
#endif

    const LlaisysRuntimeAPI *runtime_api = llaisys::core::context().runtime().api();

    const size_t logits_bytes = logits->numel() * logits->elementSize();
    std::vector<std::byte> host_logits(logits_bytes);
    runtime_api->memcpy_sync(
        host_logits.data(),
        logits->data(),
        logits_bytes,
        LLAISYS_MEMCPY_D2H);

    int64_t sampled_idx = 0;
    cpu::random_sample(
        reinterpret_cast<std::byte *>(&sampled_idx),
        host_logits.data(),
        logits->dtype(),
        logits->numel(),
        temperature,
        top_k,
        top_p,
        seed);

    runtime_api->memcpy_sync(
        out_idx->data(),
        &sampled_idx,
        sizeof(sampled_idx),
        LLAISYS_MEMCPY_H2D);
}

} // namespace llaisys::ops
