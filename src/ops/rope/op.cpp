#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/rope_nvidia.hpp"
#endif

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype()); 
    // 吸取上一步的教训：严格遵守底层协议，位置 ID 使用 I64
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "RoPE: pos_ids must be I64.");
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(), "RoPE: all tensors must be contiguous.");
    
    // 输入张量的典型形状是 [seqlen, nhead, head_dim]
    size_t seqlen = in->shape()[0];
    size_t nhead = in->shape()[1];
    size_t head_dim = in->shape()[2];

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(), out->dtype(), seqlen, nhead, head_dim, theta);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(), out->dtype(), seqlen, nhead, head_dim, theta);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rope(out->data(), in->data(), pos_ids->data(), out->dtype(), seqlen, nhead, head_dim, theta);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops