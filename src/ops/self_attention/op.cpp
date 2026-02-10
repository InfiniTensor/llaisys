#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype());
    CHECK_SAME_DTYPE(q->dtype(), k->dtype());
    CHECK_SAME_DTYPE(k->dtype(), v->dtype());

    ASSERT(q->shape().size() == 3, "SelfAttention: q must be 3-D [seqlen, nhead, d].");
    ASSERT(k->shape().size() == 3, "SelfAttention: k must be 3-D [total_len, nkvhead, d].");
    ASSERT(v->shape().size() == 3, "SelfAttention: v must be 3-D [total_len, nkvhead, dv].");
    ASSERT(attn_val->shape().size() == 3, "SelfAttention: attn_val must be 3-D [seqlen, nhead, dv].");

    size_t nh = q->shape()[1];
    size_t nh_kv = k->shape()[1];
    size_t d = q->shape()[2];

    // GQA Check
    ASSERT(nh % nh_kv == 0, "SelfAttention: nhead must be divisible by nkvhead (GQA constraint).");
    ASSERT(k->shape()[2] == d, "SelfAttention: Q and K head_dim mismatch.");
    ASSERT(attn_val->shape()[0] == q->shape()[0], "SelfAttention: Output seqlen mismatch.");
    ASSERT(attn_val->shape()[1] == nh, "SelfAttention: Output nhead mismatch.");
    ASSERT(attn_val->shape()[2] == v->shape()[2], "SelfAttention: Output head_dim mismatch with V.");

    ASSERT(q->isContiguous() && k->isContiguous() && v->isContiguous() && attn_val->isContiguous(), 
           "SelfAttention: Inputs must be contiguous.");

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), scale, attn_val->dtype(), q->shape(), k->shape(), v->shape());
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());
    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
         return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), scale, attn_val->dtype(), q->shape(), k->shape(), v->shape());
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
