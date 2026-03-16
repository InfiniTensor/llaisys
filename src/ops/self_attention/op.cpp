#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "cuda/self_attention_cuda.cuh"
#endif

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    ASSERT(q->ndim() == 3 && k->ndim() == 3 && v->ndim() == 3,
           "SelfAttention: q, k, v must be 3D [seqlen, nhead, d].");
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "SelfAttention: tensors must be contiguous.");

    size_t qlen = q->shape()[0];
    size_t nh = q->shape()[1];
    size_t d = q->shape()[2];
    size_t kvlen = k->shape()[0];
    size_t nkvh = k->shape()[1];

    if (q->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                   scale, q->dtype(), qlen, kvlen, nh, nkvh, d);
    }

    llaisys::core::context().setDevice(q->deviceType(), q->deviceId());

    switch (q->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                   scale, q->dtype(), qlen, kvlen, nh, nkvh, d);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return cuda::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                    scale, q->dtype(), qlen, kvlen, nh, nkvh, d);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
