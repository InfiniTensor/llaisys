#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/rope_cpu.hpp"
#include "nvidia/rope_nvidia.hpp"
namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    ASSERT(out->isContiguous() && in->isContiguous() 
    && pos_ids->isContiguous(), "All tensors must be Contiguous");
    ASSERT(out->shape() == in->shape(), 
    "Output shape must be equal to Input shape");
    ASSERT(pos_ids->shape()[0] == out->shape()[0], 
    "Position IDs shape[0] must be equal to Output shape[0]");
    size_t seq_len = out->shape()[0];
    size_t n_heads = out->shape()[1];
    size_t head_dim = out->shape()[2];

    if (pos_ids->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), 
        pos_ids->data(), theta, out->dtype(), seq_len, n_heads, head_dim);
    }

    llaisys::core::context().setDevice(pos_ids->deviceType(), pos_ids->deviceId());

    switch (pos_ids->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), 
        pos_ids->data(), theta, out->dtype(), seq_len, n_heads, head_dim);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rope(out->data(), in->data(), pos_ids->data(), theta, out->dtype(), seq_len, n_heads, head_dim);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
