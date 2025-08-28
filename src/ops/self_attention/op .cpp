#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {

void self_attention(tensor_t out, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(out, q, k, v);

    ASSERT(q->ndim() == 3, "self_attention: q must be [seqlen, nhead, d]");
    ASSERT(k->ndim() == 3, "self_attention: k must be [total_len, nkvhead, d]");
    ASSERT(v->ndim() == 3, "self_attention: v must be [total_len, nkvhead, dv]");
    ASSERT(out->ndim() == 3, "self_attention: out must be [seqlen, nhead, dv]");

    const size_t seqlen = q->shape()[0];
    const size_t nhead = q->shape()[1];
    const size_t d = q->shape()[2];

    const size_t total_len = k->shape()[0];
    const size_t nkvhead = k->shape()[1];
    const size_t dk = k->shape()[2];

    const size_t total_len_v = v->shape()[0];
    const size_t nkvhead_v = v->shape()[1];
    const size_t dv = v->shape()[2];

    ASSERT(d == dk, "self_attention: q.d must equal k.d");
    ASSERT(total_len == total_len_v, "self_attention: k and v total_len mismatch");
    ASSERT(nkvhead == nkvhead_v, "self_attention: nkvhead mismatch");
    ASSERT(out->shape()[0] == seqlen && out->shape()[1] == nhead && out->shape()[2] == dv,
           "self_attention: out shape must be [seqlen, nhead, dv]");
    ASSERT(total_len >= seqlen, "self_attention: total_len must be >= seqlen");

    //GQA
    ASSERT(nhead % nkvhead == 0, "self_attention: nhead must be divisible by nkvhead");

    //dtype
    ASSERT(out->dtype() == q->dtype() && out->dtype() == k->dtype() && out->dtype() == v->dtype(),
           "self_attention: all dtypes must match");
    ASSERT(out->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "self_attention: all tensors must be contiguous");

    // CPU
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(out->data(), q->data(), k->data(), v->data(),
                                   out->dtype(),
                                   seqlen, nhead, d, total_len, nkvhead, dv,
                                   scale);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(out->data(), q->data(), k->data(), v->data(),
                                   out->dtype(),
                                   seqlen, nhead, d, total_len, nkvhead, dv,
                                   scale);
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
