#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/argmax_cpu.hpp"

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    // Check that tensors are on same device
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    
    // Check data types
    ASSERT(max_idx->dtype() == LLAISYS_DTYPE_I64, "Argmax: max_idx must be int64");
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());
    
    // Check contiguity
    ASSERT(max_idx->isContiguous() && max_val->isContiguous() && vals->isContiguous(), 
           "Argmax: all tensors must be contiguous");
    
    // For now, assume vals is 1D and results are scalar (single element)
    ASSERT(vals->ndim() == 1, "Argmax: vals must be 1D tensor for now");
    ASSERT(max_idx->numel() == 1, "Argmax: max_idx must be scalar");
    ASSERT(max_val->numel() == 1, "Argmax: max_val must be scalar");

    // always support cpu calculation
    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
    }

    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());

    switch (vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
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
