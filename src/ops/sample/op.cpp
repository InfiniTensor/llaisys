#include "op.hpp"
#include "../../utils.hpp"
#include "cpu/sample_cpu.hpp"
#include "llaisys.h"
#ifdef ENABLE_NVIDIA_API
// #include "nvidia/sample_cu.cuh"  // TODO: add CUDA implementation
#endif

namespace llaisys::ops {

void sample(tensor_t out, tensor_t logits, int top_k, float top_p, float temperature) {
    ASSERT(out->numel() == 1, "sample(): out must have exactly one element");
    ASSERT(out->dtype() == LLAISYS_DTYPE_I64, "sample(): out must be int64");
    ASSERT(logits->isContiguous(), "sample(): logits must be contiguous");
    ASSERT(temperature > 0.0f, "sample(): temperature must be > 0");
    ASSERT(top_p > 0.0f && top_p <= 1.0f, "sample(): top_p must be in (0, 1]");

    switch (logits->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        cpu::sample(out->data(), logits->data(), logits->numel(),
                    top_k, top_p, temperature, logits->dtype());
        break;
#ifdef ENABLE_NVIDIA_API
    // case LLAISYS_DEVICE_NVIDIA:
    //     nvidia::sample(...);
    //     break;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

void sample_set_seed(uint64_t seed) {
    cpu::sample_set_seed(seed);
}

} // namespace llaisys::ops
