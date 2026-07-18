#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/self_attention_nvidia.cuh"
#elif defined(ENABLE_METAX_API)
#include "metax/self_attention_metax.cuh"
#endif

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "Self attention: all tensors must be contiguous.");
    ASSERT(attn_val->ndim() == 3 && q->ndim() == 3 && k->ndim() == 3 && v->ndim() == 3,
           "Self attention: all tensors must be 3D.");

    size_t seq_len = q->shape()[0];
    size_t nhead = q->shape()[1];
    size_t d = q->shape()[2];
    size_t total_len = k->shape()[0];
    size_t nkvhead = k->shape()[1];
    size_t dv = v->shape()[2];
    ASSERT(attn_val->shape()[0] == seq_len, "Self attention: output seq_len must match q.");
    ASSERT(attn_val->shape()[1] == nhead, "Self attention: output nhead must match q.");
    ASSERT(k->shape()[0] == v->shape()[0], "Self attention: k and v total_len must match.");
    ASSERT(k->shape()[1] == v->shape()[1], "Self attention: k and v nhead must match.");
    ASSERT(d == k->shape()[2], "Self attention: q and k head_dim must match.");
    ASSERT(dv == attn_val->shape()[2], "Self attention: output value dim must match v.");
    ASSERT(nkvhead > 0 && nhead % nkvhead == 0, "Self attention: nhead must be divisible by nkvhead.");

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), scale, 
                                   attn_val->dtype(), seq_len, total_len, nhead, nkvhead, d, dv);
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), scale, 
                                   attn_val->dtype(), seq_len, total_len, nhead, nkvhead, d, dv);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::self_attention(attn_val->data(), q->data(), k->data(), v->data(), scale,
                                      attn_val->dtype(), seq_len, total_len, nhead, nkvhead, d, dv);
#elif defined(ENABLE_METAX_API)
    case LLAISYS_DEVICE_METAX:
        return metax::self_attention(attn_val->data(), q->data(), k->data(), v->data(), scale,
                                     attn_val->dtype(), seq_len, total_len, nhead, nkvhead, d, dv);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
