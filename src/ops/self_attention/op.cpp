#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    // Check that tensors are on same device
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    
    // Check data types
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    
    // Check contiguity
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(), 
           "Self-Attention: all tensors must be contiguous");
    
    // Check dimensions
    ASSERT(q->ndim() == 3, "Self-Attention: q must be 3D tensor [qlen, nh, hd]");
    ASSERT(k->ndim() == 3, "Self-Attention: k must be 3D tensor [kvlen, nkvh, hd]");
    ASSERT(v->ndim() == 3, "Self-Attention: v must be 3D tensor [kvlen, nkvh, hd]");
    ASSERT(attn_val->ndim() == 3, "Self-Attention: attn_val must be 3D tensor [qlen, nh, hd]");
    
    // Get dimensions
    size_t qlen = q->shape()[0];
    size_t nh = q->shape()[1];       // number of query heads
    size_t hd = q->shape()[2];       // head dimension
    
    size_t kvlen = k->shape()[0];    // key/value sequence length
    size_t nkvh = k->shape()[1];     // number of key/value heads
    
    // Check shapes are compatible
    ASSERT(attn_val->shape()[0] == qlen, "Self-Attention: attn_val seq_len must match q");
    ASSERT(attn_val->shape()[1] == nh, "Self-Attention: attn_val heads must match q");
    ASSERT(attn_val->shape()[2] == hd, "Self-Attention: attn_val head_dim must match q");
    
    ASSERT(k->shape()[2] == hd, "Self-Attention: k head_dim must match q");
    ASSERT(v->shape()[0] == kvlen, "Self-Attention: v seq_len must match k");
    ASSERT(v->shape()[1] == nkvh, "Self-Attention: v heads must match k");
    ASSERT(v->shape()[2] == hd, "Self-Attention: v head_dim must match q");
    
    // Check Group Query Attention compatibility
    ASSERT(nh % nkvh == 0, "Self-Attention: query heads must be divisible by key/value heads");

    // always support cpu calculation
    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                  attn_val->dtype(), qlen, kvlen, nh, nkvh, hd, scale);
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                  attn_val->dtype(), qlen, kvlen, nh, nkvh, hd, scale);
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
