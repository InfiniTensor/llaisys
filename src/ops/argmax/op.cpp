#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/argmax_cpu.hpp"
#include "nvidia/argmax_nvidia.hpp"

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    ASSERT(vals->ndim() == 1, "Argmax: vals must be 1D.");
    ASSERT(max_idx->ndim() == 1 && max_idx->shape()[0] == 1, "Argmax: max_idx shape must be (1, ).");
    ASSERT(max_val->ndim() == 1 && max_val->shape()[0] == 1, "Argmax: max_val shape must be (1, ).");
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());
    ASSERT(max_idx->dtype() == LLAISYS_DTYPE_I64, "Argmax: max_idx must be I64.");
    ASSERT(vals->isContiguous() && max_idx->isContiguous() && max_val->isContiguous(), "Argmax: all tensors must be contiguous.");

    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
    }

    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());

    switch (vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
