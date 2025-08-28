#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/argmax_cpu.hpp"

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);

    ASSERT(vals->ndim() == 1, "Argmax: vals must be 1D for now");
    ASSERT(max_idx->ndim() == 1 && max_val->ndim() == 1,
           "Argmax: outputs must be 1D");
    ASSERT(max_idx->numel() == 1 && max_val->numel() == 1,
           "Argmax: outputs must have a single element");
    ASSERT(vals->isContiguous(), "Argmax: vals must be contiguous");
    ASSERT(max_idx->isContiguous() && max_val->isContiguous(),
           "Argmax: outputs must be contiguous");

    ASSERT(max_idx->dtype() == LLAISYS_DTYPE_I64,
           "Argmax: max_idx must be int64");
    ASSERT(max_val->dtype() == vals->dtype(),
           "Argmax: max_val dtype must match vals dtype");

    // --- Dispatch ---
    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(max_idx->data(),
                           max_val->data(),
                           vals->data(),
                           vals->dtype(),
                           vals->numel());
    }

    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());

    switch (vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::argmax(max_idx->data(),
                           max_val->data(),
                           vals->data(),
                           vals->dtype(),
                           vals->numel());
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
