#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/argmax_cpu.hpp"

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    // Only support contiguous inputs for now.
    ASSERT(max_idx->isContiguous() && max_val->isContiguous() && vals->isContiguous(), "Argmax: all tensors must be contiguous.");
    /* Deprecated
    Tests use i64.
    // // Data of max_idx should be of type size_t
    // // size_t 应该是 U32 或者 U64 吧...
    // ASSERT(max_idx->dtype() == LLAISYS_DTYPE_U32 || max_idx->dtype() == LLAISYS_DTYPE_U64,
    //        "Argmax: max_idx should in type U32 or U64");
    // Data of max_val and vals should be in same type
    */

    // 假设 idx_type 永远是 LLAISYS_DTYPE_I64
    ASSERT(max_idx->dtype() == LLAISYS_DTYPE_I64, "Argmax: idx_type must be LLAISYS_DTYPE_I64");

    ASSERT(max_val->dtype() == vals->dtype(), "Argmax: max_val and vals should have be of same type");

    ASSERT(vals->numel(), "Argmax: the tensor must be non-empty");

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
