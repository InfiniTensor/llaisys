#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/argmax_cpu.hpp"

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);

    ASSERT(vals->isContiguous() && max_idx->isContiguous() && max_val->isContiguous(),
           "Argmax: all tensors must be contiguous.");

    ASSERT(vals->ndim() == 1, "Argmax: input 'vals' must be a 1D tensor.");

    ASSERT(max_val->numel() == 1 && max_idx->numel() == 1,
           "Argmax: output tensors must be scalars (single element).");

    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());

    ASSERT(max_idx->dtype() == LLAISYS_DTYPE_I64,
           "Argmax: 'max_idx' must be Int64.");

    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(vals->data(), max_val->data(), max_idx->data(), vals->dtype(), vals->numel());
    }

    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());

    switch (vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:

        return cpu::argmax(vals->data(), max_val->data(), max_idx->data(), vals->dtype(), vals->numel());
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
