#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());

    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "RoPE: pos_ids must be int64.");
    
    ASSERT(in->shape().size() == 3, "RoPE: input tensor must be 3-D [seqlen, nhead, head_dim].");
    ASSERT(out->shape().size() == 3, "RoPE: output tensor must be 3-D.");
    ASSERT(pos_ids->shape().size() == 1, "RoPE: pos_ids tensor must be 1-D [seqlen].");

    size_t seq_len = in->shape()[0];
    size_t head_dim = in->shape()[2];

    ASSERT(pos_ids->shape()[0] == seq_len, "RoPE: pos_ids length mismatch with input seqlen.");
    ASSERT(out->shape() == in->shape(), "RoPE: output shape mismatch with input.");
    ASSERT(head_dim % 2 == 0, "RoPE: head_dim must be even.");

    ASSERT(in->isContiguous() && out->isContiguous() && pos_ids->isContiguous(), "RoPE: inputs must be contiguous.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(), theta, out->dtype(), in->shape());
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(), theta, out->dtype(), in->shape());
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
