#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/rope_cpu.hpp" 

namespace llaisys::ops {

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {

    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());

    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "RoPE: pos_ids must be Int64.");

    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
           "RoPE: all tensors must be contiguous.");

    ASSERT(in->ndim() == 3, "RoPE: input must be 3D [seq, head, dim].");
    ASSERT(out->ndim() == 3, "RoPE: output must be 3D [seq, head, dim].");

    ASSERT(pos_ids->ndim() == 1, "RoPE: pos_ids must be 1D [seq].");

    ASSERT(in->shape()[0] == out->shape()[0], "RoPE: input/output seq_len mismatch.");
    ASSERT(in->shape()[1] == out->shape()[1], "RoPE: input/output n_heads mismatch.");
    ASSERT(in->shape()[2] == out->shape()[2], "RoPE: input/output head_dim mismatch.");

    ASSERT(pos_ids->shape()[0] == in->shape()[0], "RoPE: pos_ids length must match seq_len.");

    size_t head_dim = in->shape()[2];
    ASSERT(head_dim % 2 == 0, "RoPE: head_dim must be even.");

    size_t seq_len = in->shape()[0];
    size_t n_heads = in->shape()[1];

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        ::llaisys::ops::cpu::rope(
            out->data(),
            in->data(),
            pos_ids->data(),
            out->dtype(),
            seq_len,
            n_heads,
            head_dim,
            theta);
        return;
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        ::llaisys::ops::cpu::rope(
            out->data(), in->data(), pos_ids->data(),
            out->dtype(), seq_len, n_heads, head_dim, theta);
        return;

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
