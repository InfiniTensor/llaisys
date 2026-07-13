#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/rope_nvidia.cuh"
#endif

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(), "RoPE: all tensors must be contiguous.");
    ASSERT(out->shape() == in->shape(), "RoPE: out and in must have same shape.");
    ASSERT(out->ndim() == 3, "RoPE: out must be 3D tensor [seq_len, n_heads, head_dim].");
    ASSERT(in->ndim() == 3, "RoPE: in must be 3D tensor [seq_len, n_heads, head_dim].");
    ASSERT(pos_ids->ndim() == 1, "RoPE: pos_ids must be 1D tensor [seq_len].");
    ASSERT(pos_ids->shape()[0] == out->shape()[0], "RoPE: pos_ids length must match seq_len.");
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "RoPE: pos_ids must be int64.");
    ASSERT(out->shape()[2] % 2 == 0, "RoPE: head_dim must be even.");

    size_t seq_len = out->shape()[0];
    size_t n_heads = out->shape()[1];
    size_t head_dim = out->shape()[2];

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), reinterpret_cast<const int64_t*>(pos_ids->data()), theta, out->dtype(), seq_len, n_heads, head_dim);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), reinterpret_cast<const int64_t*>(pos_ids->data()), theta, out->dtype(), seq_len, n_heads, head_dim);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rope(out->data(), in->data(), reinterpret_cast<const int64_t *>(pos_ids->data()), theta, out->dtype(), seq_len,
                            n_heads, head_dim, llaisys::core::context().runtime().stream());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
