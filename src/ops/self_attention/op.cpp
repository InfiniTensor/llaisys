#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/self_attention_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/self_attention_metax.hpp"
#endif

// Q: [seqlen, nhead, d], K: [total_len, nkvhead, d], V: [total_len, nkvhead, dv], attn_val: [seqlen, nhead, dv]
namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    // 1. 参数校验
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    CHECK_ARGUMENT(attn_val->ndim() == 3 && q->ndim() == 3 && k->ndim() == 3 && v->ndim() == 3,
                  "self_attention: all tensors must be 3D");
    CHECK_ARGUMENT(attn_val->shape()[0] == q->shape()[0], "self_attention: seqlen of attn_val and q must match");
    CHECK_ARGUMENT(attn_val->shape()[1] == q->shape()[1], "self_attention: nhead of attn_val and q must match");
    CHECK_ARGUMENT(q->shape()[2] == k->shape()[2], "self_attention: d of q and k must match");
    CHECK_ARGUMENT(attn_val->shape()[2] == v->shape()[2], "self_attention: dv of attn_val and v must match");
    CHECK_ARGUMENT(k->shape()[0] == v->shape()[0] && k->shape()[1] == v->shape()[1],
                  "self_attention: total_len and nkvhead of k and v must match");
    CHECK_ARGUMENT((q->shape()[1] % k->shape()[1]) == 0, "self_attention: nhead must be divisible by nkvhead (GQA)");
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "self_attention: all tensors must be contiguous");

    const size_t seqlen = q->shape()[0];
    const size_t nhead = q->shape()[1];
    const size_t d = q->shape()[2];
    const size_t total_len = k->shape()[0];
    const size_t nkvhead = k->shape()[1];
    const size_t dv = v->shape()[2];

    // 2. 设置设备上下文
    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    // 3. 设备分发
    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                   attn_val->dtype(), seqlen, nhead, nkvhead, d, dv, total_len, scale);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::self_attention(attn_val->data(),
                                      q->data(),
                                      k->data(),
                                      v->data(),
                                      attn_val->dtype(),
                                      seqlen,
                                      nhead,
                                      nkvhead,
                                      d,
                                      dv,
                                      total_len,
                                      scale);
#endif
#ifdef ENABLE_METAX_API
    case LLAISYS_DEVICE_METAX:
        return metax::self_attention(attn_val->data(),
                                     q->data(),
                                     k->data(),
                                     v->data(),
                                     attn_val->dtype(),
                                     seqlen,
                                     nhead,
                                     nkvhead,
                                     d,
                                     dv,
                                     total_len,
                                     scale);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
