#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    // Only support contiguous inputs for now.
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(), "RoPE: all tensors must be contiguous.");
    ASSERT(out->ndim() == 3, "RoPE: out dimension must be 3");
    ASSERT(pos_ids->ndim() == 1, "RoPE: pos_ids dimension must be 1");
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "RoPE: pos_ids must be int64");
    size_t head_dim = in->shape()[2];
    ASSERT(head_dim % 2 == 0, "RoPE: head dimension must be even");
    ASSERT(pos_ids->numel() == in->shape()[0], "RoPE: pos_ids length must match seq_len");

    // always support cpu calculation
    if (in->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(), in->dtype(),
                         in->shape()[0], in->shape()[1], in->shape()[2], theta);
    }

    llaisys::core::context().setDevice(in->deviceType(), in->deviceId());

    switch (in->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(), in->dtype(),
                         in->shape()[0], in->shape()[1], in->shape()[2], theta);
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
