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

    // Q: [seqlen, nhead, d]
    // K: [total_len, nkvhead, d]
    // V: [total_len, nkvhead, dv]
    // Out: [seqlen, nhead, dv]
    ASSERT(q->ndim() == 3, "SelfAttention: q must be 3D.");
    ASSERT(k->ndim() == 3, "SelfAttention: k must be 3D.");
    ASSERT(v->ndim() == 3, "SelfAttention: v must be 3D.");
    ASSERT(attn_val->ndim() == 3, "SelfAttention: attn_val must be 3D.");

    size_t seqlen = q->shape()[0];
    size_t nhead = q->shape()[1];
    size_t head_dim = q->shape()[2]; // d

    size_t total_len = k->shape()[0];
    size_t nkvhead = k->shape()[1];

    size_t head_dim_v = v->shape()[2]; // dv

    ASSERT(nhead % nkvhead == 0, "SelfAttention: nhead must be divisible by nkvhead (GQA).");

    // 检查维度匹配
    ASSERT(k->shape()[2] == head_dim, "SelfAttention: k head_dim must match q head_dim.");
    ASSERT(v->shape()[0] == total_len, "SelfAttention: v total_len must match k total_len.");
    ASSERT(v->shape()[1] == nkvhead, "SelfAttention: v nkvhead must match k nkvhead.");

    // 检查输出维度
    ASSERT(attn_val->shape()[0] == seqlen, "SelfAttention: output seqlen mismatch.");
    ASSERT(attn_val->shape()[1] == nhead, "SelfAttention: output nhead mismatch.");
    ASSERT(attn_val->shape()[2] == head_dim_v, "SelfAttention: output head_dim_v mismatch.");

    if (q->deviceType() == LLAISYS_DEVICE_CPU) {
        ::llaisys::ops::cpu::self_attention(
            q->data(),
            k->data(),
            v->data(),
            attn_val->data(),
            q->dtype(),
            seqlen,
            total_len,
            nhead,
            nkvhead,
            head_dim,
            head_dim_v,
            scale);
        return;
    }

    llaisys::core::context().setDevice(q->deviceType(), q->deviceId());

    switch (q->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        ::llaisys::ops::cpu::self_attention(
            q->data(), k->data(), v->data(), attn_val->data(),
            q->dtype(), seqlen, total_len, nhead, nkvhead,
            head_dim, head_dim_v, scale);
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
