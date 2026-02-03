#include "op.hpp"
#include "../../utils.hpp"
#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());

    ASSERT(attn_val->ndim() == 3 && q->ndim() == 3 && k->ndim() == 3 && v->ndim() == 3,
           "self_attention: all tensors must be 3D");
    const auto &qs = q->shape();      // [qlen, nhead, head_dim]
    const auto &ks = k->shape();      // [kvlen, nkvhead, head_dim]
    const auto &vs = v->shape();      // [kvlen, nkvhead, value_dim]
    const auto &os = attn_val->shape(); // [qlen, nhead, value_dim]

    size_t qlen = qs[0], nhead = qs[1], head_dim = qs[2];
    size_t kvlen = ks[0], nkvhead = ks[1], k_head_dim = ks[2];
    size_t value_dim = vs[2];

    ASSERT(head_dim == k_head_dim, "q/k head_dim must match");
    ASSERT(vs[0] == kvlen && vs[1] == nkvhead, "v shape must align with k");
    ASSERT(os[0] == qlen && os[1] == nhead && os[2] == value_dim, "attn_val shape mismatch");
    ASSERT(nhead % nkvhead == 0, "nhead must be a multiple of nkvhead for head repeat");
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "self_attention: all tensors must be contiguous");

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        cpu::self_attention(
            attn_val->data(),
            q->data(),
            k->data(),
            v->data(),
            attn_val->dtype(),
            qlen,
            kvlen,
            nhead,
            nkvhead,
            head_dim,
            value_dim,
            scale);
        return;
    }

    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
