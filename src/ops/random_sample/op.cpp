#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/random_sample_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "cuda/random_sample_cuda.cuh"
#endif

namespace llaisys::ops {
void random_sample(tensor_t sample_idx, tensor_t sample_val, tensor_t logits, float temperature, int top_k, float top_p,
                   int64_t seed) {
    CHECK_SAME_DEVICE(sample_idx, sample_val, logits);
    CHECK_SAME_DTYPE(sample_val->dtype(), logits->dtype());

    ASSERT(logits->ndim() == 1, "RandomSample: logits must be a 1D tensor");
    ASSERT(logits->numel() > 0, "RandomSample: logits cannot be empty");
    ASSERT(sample_idx->ndim() == 1 && sample_idx->shape()[0] == 1,
           "RandomSample: sample_idx must be a 1D tensor with single element");
    ASSERT(sample_val->ndim() == 1 && sample_val->shape()[0] == 1,
           "RandomSample: sample_val must be a 1D tensor with single element");

    ASSERT(logits->isContiguous() && sample_idx->isContiguous() && sample_val->isContiguous(),
           "RandomSample: tensors must be contiguous");
    ASSERT(sample_idx->dtype() == LLAISYS_DTYPE_I64, "RandomSample: sample_idx must be Int64 type");
    ASSERT(logits->dtype() == LLAISYS_DTYPE_F32 || logits->dtype() == LLAISYS_DTYPE_F16 || logits->dtype() == LLAISYS_DTYPE_BF16,
           "RandomSample: logits dtype must be Float32/Float16/BFloat16");

    if (logits->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::random_sample(sample_idx->data(), sample_val->data(), logits->data(), logits->dtype(), logits->numel(),
                                  temperature, top_k, top_p, seed);
    }

    llaisys::core::context().setDevice(logits->deviceType(), logits->deviceId());

    switch (logits->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::random_sample(sample_idx->data(), sample_val->data(), logits->data(), logits->dtype(), logits->numel(),
                                  temperature, top_k, top_p, seed);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return cuda::random_sample(sample_idx->data(), sample_val->data(), logits->data(), logits->dtype(), logits->numel(),
                                   temperature, top_k, top_p, seed);
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
