#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/rope_nvidia.cuh"
#elif defined(ENABLE_METAX_API)
#include "metax/rope_metax.cuh"
#endif

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(), "RoPE: inputs must be contiguous.");
    ASSERT(in->ndim() == 3, "RoPE: input tensor must be 3D.");
    ASSERT(out->ndim() == 3, "RoPE: output tensor must be 3D.");
    ASSERT(pos_ids->ndim() == 1, "RoPE: pos_ids tensor must be 1D.");
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "RoPE: pos_ids tensor must be int64.");
    ASSERT(out->shape() == in->shape(), "RoPE: output shape must match input shape.");

    size_t seq_len = in->shape()[0];
    size_t n_heads = in->shape()[1];
    size_t head_dim = in->shape()[2];
    ASSERT(pos_ids->shape()[0] == seq_len, "RoPE: pos_ids length must match sequence length.");
    ASSERT(head_dim % 2 == 0, "RoPE: head dimension must be even.");

    if (in->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(),
                        theta, in->dtype(), seq_len, n_heads, head_dim);
    }

    llaisys::core::context().setDevice(in->deviceType(), in->deviceId());

    switch (in->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(),
                        theta, in->dtype(), seq_len, n_heads, head_dim);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rope(out->data(), in->data(), pos_ids->data(),
                            theta, in->dtype(), seq_len, n_heads, head_dim);
#elif defined(ENABLE_METAX_API)
    case LLAISYS_DEVICE_METAX:
        return metax::rope(out->data(), in->data(), pos_ids->data(),
                           theta, in->dtype(), seq_len, n_heads, head_dim);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
