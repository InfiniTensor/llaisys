#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/self_attention_nvidia.cuh"
#endif

#ifdef ENABLE_ILUVATAR_API
#include "iluvatar/self_attention_iluvatar.cuh"
#endif

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    
    // Check tensor dimensions
    ASSERT(q->ndim() == 3, "SelfAttention: q must be 3D tensor.");
    ASSERT(k->ndim() == 3, "SelfAttention: k must be 3D tensor.");
    ASSERT(v->ndim() == 3, "SelfAttention: v must be 3D tensor.");
    ASSERT(attn_val->ndim() == 3, "SelfAttention: attn_val must be 3D tensor.");
    
    // Check dtypes
    ASSERT(q->dtype() == k->dtype() && q->dtype() == v->dtype() && q->dtype() == attn_val->dtype(), 
           "SelfAttention: all tensors must have same dtype.");
    
    // Get dimensions
    size_t seq_len = q->shape()[0];
    size_t total_len = k->shape()[0];
    size_t nhead = q->shape()[1];
    size_t nkvhead = k->shape()[1];
    size_t d = q->shape()[2];
    size_t dv = v->shape()[2];
    
    // Check shapes
    ASSERT(k->shape()[2] == d, "SelfAttention: k shape[2] must match q shape[2].");
    ASSERT(v->shape()[0] == total_len, "SelfAttention: v shape[0] must match k shape[0].");
    ASSERT(v->shape()[1] == nkvhead, "SelfAttention: v shape[1] must match k shape[1].");
    ASSERT(attn_val->shape()[0] == seq_len, "SelfAttention: attn_val shape[0] must match q shape[0].");
    ASSERT(attn_val->shape()[1] == nhead, "SelfAttention: attn_val shape[1] must match q shape[1].");
    ASSERT(attn_val->shape()[2] == dv, "SelfAttention: attn_val shape[2] must match v shape[2].");
    ASSERT(nhead % nkvhead == 0, "SelfAttention: nhead must be divisible by nkvhead.");
    
    // Check contiguity
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(), 
           "SelfAttention: all tensors must be contiguous.");

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(
            attn_val->data(), 
            q->data(), 
            k->data(), 
            v->data(), 
            q->dtype(), 
            scale, 
            seq_len, 
            total_len, 
            nhead, 
            nkvhead, 
            d, 
            dv
        );
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(
            attn_val->data(), 
            q->data(), 
            k->data(), 
            v->data(), 
            q->dtype(), 
            scale, 
            seq_len, 
            total_len, 
            nhead, 
            nkvhead, 
            d, 
            dv
        );
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::self_attention(
            attn_val->data(), 
            q->data(), 
            k->data(), 
            v->data(), 
            q->dtype(), 
            scale, 
            seq_len, 
            total_len, 
            nhead, 
            nkvhead, 
            d, 
            dv
        );
#endif

#ifdef ENABLE_ILUVATAR_API
    case LLAISYS_DEVICE_ILUVATAR:
        return iluvatar::self_attention(
            attn_val->data(), 
            q->data(), 
            k->data(), 
            v->data(), 
            q->dtype(), 
            scale, 
            seq_len, 
            total_len, 
            nhead, 
            nkvhead, 
            d, 
            dv
        );
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops