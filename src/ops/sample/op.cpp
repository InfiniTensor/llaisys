#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../device/runtime_api.hpp"
#include "../../utils.hpp"

#include "cpu/sample_cpu.hpp"

#include <vector>

namespace llaisys::ops {
int64_t sample(tensor_t logits, float temperature, int top_k, float top_p, uint64_t seed) {
    ASSERT(logits != nullptr, "Sample: logits tensor must not be null.");
    ASSERT(logits->isContiguous(), "Sample: logits tensor must be contiguous.");

    if (logits->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::sample(logits->data(), logits->numel(), logits->dtype(), temperature, top_k, top_p, seed);
    }

    // 采样只发生在最后一步 logits 上，直接拷回 Host 做概率筛选即可。
    llaisys::core::context().setDevice(logits->deviceType(), logits->deviceId());
    auto &runtime = llaisys::core::context().runtime();
    auto host_storage = runtime.allocateHostStorage(logits->numel() * logits->elementSize());
    runtime.api()->memcpy_sync(host_storage->memory(), logits->data(), logits->numel() * logits->elementSize(), LLAISYS_MEMCPY_D2H);
    return cpu::sample(host_storage->memory(), logits->numel(), logits->dtype(), temperature, top_k, top_p, seed);
}
} // namespace llaisys::ops
