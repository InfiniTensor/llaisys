#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rand_sample_cpu.hpp"
#include "nvidia/rand_sample_nvidia.hpp"

namespace llaisys::ops {
void rand_sample(tensor_t sample_idx, tensor_t sample_val, tensor_t vals, float temperature, size_t topK, float topP,
                 int64_t seed) {
    CHECK_SAME_DEVICE(sample_idx, sample_val, vals);
    CHECK_SAME_DTYPE(sample_val->dtype(), vals->dtype());
    ASSERT(sample_idx->dtype() == LLAISYS_DTYPE_I64, "rand_sample: sample_idx must be I64.");
    ASSERT(sample_idx->isContiguous() && sample_val->isContiguous() && vals->isContiguous(),
           "rand_sample: all tensors must be contiguous.");

    size_t batch_size = 1;
    size_t numel = 0;
    if (vals->ndim() == 1) {
        numel = vals->shape()[0];
        ASSERT(sample_idx->ndim() == 1 && sample_idx->shape()[0] == 1, "rand_sample: sample_idx shape must be (1, ).");
        ASSERT(sample_val->ndim() == 1 && sample_val->shape()[0] == 1, "rand_sample: sample_val shape must be (1, ).");
    } else {
        ASSERT(vals->ndim() == 2, "rand_sample: vals must be 1D or 2D.");
        batch_size = vals->shape()[0];
        numel = vals->shape()[1];
        ASSERT(sample_idx->ndim() == 1 && sample_idx->shape()[0] == batch_size,
               "rand_sample: sample_idx shape must be (batch, ).");
        ASSERT(sample_val->ndim() == 1 && sample_val->shape()[0] == batch_size,
               "rand_sample: sample_val shape must be (batch, ).");
    }

    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rand_sample(sample_idx->data(), sample_val->data(), vals->data(), vals->dtype(), numel,
                                static_cast<int64_t>(batch_size), temperature, topK, topP, seed);
    }

    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());

    switch (vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rand_sample(sample_idx->data(), sample_val->data(), vals->data(), vals->dtype(), numel,
                                static_cast<int64_t>(batch_size), temperature, topK, topP, seed);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rand_sample(sample_idx->data(), sample_val->data(), vals->data(), vals->dtype(), numel,
                                   static_cast<int64_t>(batch_size), temperature, topK, topP, seed);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
