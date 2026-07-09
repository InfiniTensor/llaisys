#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/argmax_cpu.hpp"

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    // Check all tensors on same device
    CHECK_SAME_DEVICE(max_idx, max_val, vals);

    // vals should be 1D, max_idx and max_val should have 1 element
    ASSERT(vals->ndim() == 1, "vals must be 1D tensor");
    ASSERT(max_idx->numel() == 1, "max_idx must have 1 element");
    ASSERT(max_val->numel() == 1, "max_val must have 1 element");
    ASSERT(max_val->dtype() == vals->dtype(), "max_val and vals must have same dtype");

    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());

    switch (vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(),
                          max_idx->dtype(), vals->dtype(), vals->numel());
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
