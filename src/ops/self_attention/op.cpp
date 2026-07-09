#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/self_attention_cpu.hpp"
#include "nvidia/self_attention_nvidia.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    ASSERT(attn_val->ndim() == 3 && q->ndim() == 3 && k->ndim() == 3 && v->ndim() == 3,
           "self_attention: all tensors must be 3D.");
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "self_attention: all tensors must be contiguous.");
    ASSERT(q->shape()[0] == attn_val->shape()[0], "self_attention: qlen mismatch.");
    ASSERT(q->shape()[1] == attn_val->shape()[1], "self_attention: nhead mismatch.");
    ASSERT(v->shape()[2] == attn_val->shape()[2], "self_attention: dv mismatch.");
    ASSERT(q->shape()[2] == k->shape()[2], "self_attention: q/k head dim mismatch.");
    ASSERT(k->shape()[0] == v->shape()[0], "self_attention: k/v length mismatch.");
    ASSERT(k->shape()[1] == v->shape()[1], "self_attention: k/v head mismatch.");
    ASSERT(q->shape()[1] % k->shape()[1] == 0, "self_attention: nhead must be multiple of nkvhead.");

    size_t qlen = q->shape()[0];
    size_t kvlen = k->shape()[0];
    size_t nh = q->shape()[1];
    size_t nkvh = k->shape()[1];
    size_t hd = q->shape()[2];
    size_t dv = v->shape()[2];

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), scale,
                                   attn_val->dtype(), qlen, kvlen, nh, nkvh, hd, dv);
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), scale,
                                   attn_val->dtype(), qlen, kvlen, nh, nkvh, hd, dv);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::self_attention(attn_val->data(), q->data(), k->data(), v->data(), scale, attn_val->dtype(), qlen,
                                      kvlen, nh, nkvh, hd, dv);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
