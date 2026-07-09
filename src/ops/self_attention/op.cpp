#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    // Only support contiguous inputs.
    CHECK_SAME_SHAPE(attn_val->shape()[0], q->shape()[0]);
    CHECK_SAME_SHAPE(attn_val->shape()[1], q->shape()[1]);
    CHECK_SAME_SHAPE(attn_val->shape()[2], v->shape()[2]);
    CHECK_SAME_SHAPE(k->shape()[0], v->shape()[0]);
    CHECK_SAME_SHAPE(k->shape()[1], v->shape()[1]);
    CHECK_SAME_SHAPE(q->shape()[2], k->shape()[2]);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(), "Self-Attention: all tensors must be contiguous.");

    const size_t seqlen = attn_val->shape()[0];
    const size_t nhead = attn_val->shape()[1];
    // FIX: d should be the head dimension of Q/K, not the output (which is dv)
    const size_t d = q->shape()[2]; 
    const size_t token_len = k->shape()[0];
    const size_t nkvhead = k->shape()[1];
    const size_t dv = v->shape()[2];

    // always support cpu calculation
    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), attn_val->dtype(), seqlen, nhead, nkvhead, d, dv, token_len, scale);
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), attn_val->dtype(), seqlen, nhead, nkvhead, d, dv, token_len, scale);
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
