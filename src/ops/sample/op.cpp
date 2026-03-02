#include "op.hpp"
#include "../../utils.hpp"
#include "cpu/sample_cpu.hpp"

namespace llaisys::ops {
void sample(tensor_t out_idx, tensor_t logits, int top_k, float top_p, float temperature, int64_t seed) {
    CHECK_SAME_DEVICE(out_idx, logits);

    ASSERT(out_idx->dtype() == LLAISYS_DTYPE_I64, "Output tensor must be int64");
    ASSERT(out_idx->numel() == 1, "Output tensor must have a single element");
    ASSERT(logits->ndim() == 1, "Input tensor must be 1-dimensional");
    ASSERT(logits->numel() > 0, "Input tensor must not be empty");
    ASSERT(
        logits->dtype() == LLAISYS_DTYPE_F32 ||
            logits->dtype() == LLAISYS_DTYPE_F16 ||
            logits->dtype() == LLAISYS_DTYPE_BF16,
        "Input tensor data type must be F32/F16/BF16");

    ASSERT(out_idx->isContiguous(), "Output tensor must be contiguous");
    ASSERT(logits->isContiguous(), "Logits tensor must be contiguous");

    ASSERT(temperature > 0.0f, "Temperature must be greater than 0");
    // Relaxed top_p check: 0.0 means no top-p sampling
    ASSERT(top_p >= 0.0f && top_p <= 1.0f, "Top-p must be in the range [0, 1]");
    ASSERT(top_k >= 1, "Top-k must be >= 1");

    if (logits->deviceType() == LLAISYS_DEVICE_CPU)
    {
        cpu::sample(out_idx->data(), logits->data(), logits->dtype(), logits->numel(), top_k, top_p, temperature, seed);
        return;
    }

    EXCEPTION_UNSUPPORTED_DEVICE;
}
} //namespace llaisys::ops
