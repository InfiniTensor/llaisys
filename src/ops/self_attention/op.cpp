#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/self_attention_nvidia.cuh"
#endif

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "SelfAttention: all tensors must be contiguous.");
    
    // Shape checks
    // q: [qlen, nhead, head_dim]
    // k: [kvlen, nkvhead, head_dim]
    // v: [kvlen, nkvhead, head_dim]
    // attn_val: [qlen, nhead, head_dim]
    ASSERT(q->ndim() == 3, "SelfAttention: q must be 3D tensor [qlen, nhead, head_dim].");
    ASSERT(k->ndim() == 3, "SelfAttention: k must be 3D tensor [kvlen, nkvhead, head_dim].");
    ASSERT(v->ndim() == 3, "SelfAttention: v must be 3D tensor [kvlen, nkvhead, head_dim].");
    ASSERT(attn_val->ndim() == 3, "SelfAttention: attn_val must be 3D tensor [qlen, nhead, head_dim].");
    
    size_t qlen = q->shape()[0];
    size_t nhead = q->shape()[1];
    size_t head_dim = q->shape()[2];
    size_t kvlen = k->shape()[0];
    size_t nkvhead = k->shape()[1];
    
    // Validate shapes
    ASSERT(k->shape()[0] == v->shape()[0], "SelfAttention: k and v must have same kvlen.");
    ASSERT(k->shape()[1] == v->shape()[1], "SelfAttention: k and v must have same nkvhead.");
    ASSERT(k->shape()[2] == head_dim, "SelfAttention: k head_dim must match q head_dim.");
    ASSERT(v->shape()[2] == head_dim, "SelfAttention: v head_dim must match q head_dim.");
    ASSERT(attn_val->shape()[0] == qlen, "SelfAttention: attn_val qlen must match q qlen.");
    ASSERT(attn_val->shape()[1] == nhead, "SelfAttention: attn_val nhead must match q nhead.");
    ASSERT(attn_val->shape()[2] == head_dim, "SelfAttention: attn_val head_dim must match q head_dim.");
    ASSERT(nhead % nkvhead == 0, "SelfAttention: nhead must be divisible by nkvhead for GQA.");
    ASSERT(kvlen >= qlen, "SelfAttention: kvlen must be >= qlen.");
    
    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                   scale, attn_val->dtype(), qlen, kvlen, nhead, nkvhead, head_dim);
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                   scale, attn_val->dtype(), qlen, kvlen, nhead, nkvhead, head_dim);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::self_attention(attn_val->data(), q->data(), k->data(), v->data(), scale, attn_val->dtype(), qlen, kvlen,
                                      nhead, nkvhead, head_dim, llaisys::core::context().runtime().stream());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
