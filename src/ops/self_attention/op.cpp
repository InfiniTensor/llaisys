#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "SelfAttention: all tensors must be contiguous.");

    ASSERT(q->ndim() == 3 && k->ndim() == 3 && v->ndim() == 3 && attn_val->ndim() == 3,
           "SelfAttention: q/k/v/attn_val must be 3D [len, heads, head_dim]");

    const size_t qlen = q->shape()[0];
    const size_t nh = q->shape()[1];
    const size_t hd = q->shape()[2];

    const size_t kvlen = k->shape()[0];
    const size_t nkvh = k->shape()[1];

    CHECK_SAME_SHAPE(v->shape(), k->shape());
    ASSERT(attn_val->shape()[0] == qlen && attn_val->shape()[1] == nh && attn_val->shape()[2] == hd,
           "SelfAttention: attn_val shape must match q shape [qlen, nh, hd]");
    ASSERT(k->shape()[2] == hd, "SelfAttention: head_dim mismatch between q and k");
    ASSERT(nh % nkvh == 0, "SelfAttention: require nh % nkvh == 0 (GQA/MQA head mapping)");
    ASSERT(kvlen >= qlen, "SelfAttention: currently require kvlen >= qlen for causal masking");

    // always support cpu calculation
    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                   attn_val->dtype(), qlen, nh, kvlen, nkvh, hd, scale);
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                   attn_val->dtype(), qlen, nh, kvlen, nkvh, hd, scale);
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
