#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in);
    CHECK_SAME_DEVICE(out, pos_ids);
    
    // Only support 3D tensors for now: [seqlen, nhead, d] or [seqlen, nkvhead, d]
    ASSERT(in->ndim() == 3, "RoPE: in must be 3D tensor.");
    ASSERT(out->ndim() == 3, "RoPE: out must be 3D tensor.");
    ASSERT(pos_ids->ndim() == 1, "RoPE: pos_ids must be 1D tensor.");
    ASSERT(in->dtype() == out->dtype(), "RoPE: in and out must have same dtype.");
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "RoPE: pos_ids must be int64 dtype.");
    ASSERT(in->shape() == out->shape(), "RoPE: in and out must have same shape.");
    ASSERT(pos_ids->shape()[0] == in->shape()[0], "RoPE: pos_ids shape[0] must match in shape[0].");
    ASSERT(in->shape()[2] % 2 == 0, "RoPE: in shape[2] must be even.");
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(), "RoPE: all tensors must be contiguous.");

    size_t seq_len = in->shape()[0];
    size_t num_heads = in->shape()[1];
    size_t head_dim = in->shape()[2];

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(
            out->data(), 
            in->data(), 
            pos_ids->data(), 
            in->dtype(), 
            theta, 
            seq_len, 
            num_heads, 
            head_dim
        );
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(
            out->data(), 
            in->data(), 
            pos_ids->data(), 
            in->dtype(), 
            theta, 
            seq_len, 
            num_heads, 
            head_dim
        );
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