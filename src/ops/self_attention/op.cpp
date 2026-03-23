#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/self_attention_nvidia.hpp"
#endif
#ifdef ENABLE_ILUVATAR_API
#include "nvidia/self_attention_nvidia.hpp"
#endif

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());

    ASSERT(attn_val->ndim() == 3 && q->ndim() == 3 && k->ndim() == 3 && v->ndim() == 3,
           "SelfAttention: all tensors must be 3D.");

    size_t qlen = q->shape()[0];
    size_t nhead = q->shape()[1];
    size_t dim = q->shape()[2];

    size_t kvlen = k->shape()[0];
    size_t nkvh = k->shape()[1];
    size_t kdim = k->shape()[2];
    size_t vdim = v->shape()[2];

    ASSERT(dim == kdim, "SelfAttention: q and k head dim mismatch.");
    ASSERT(v->shape()[0] == kvlen && v->shape()[1] == nkvh, "SelfAttention: v shape mismatch with k.");
    ASSERT(attn_val->shape()[0] == qlen && attn_val->shape()[1] == nhead && attn_val->shape()[2] == vdim,
           "SelfAttention: output shape mismatch.");
    ASSERT(nhead % nkvh == 0, "SelfAttention: nhead must be divisible by nkvh.");

    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "SelfAttention: tensors must be contiguous.");

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), attn_val->dtype(), qlen,
                                   kvlen, nhead, nkvh, dim, vdim, scale);
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), attn_val->dtype(), qlen,
                                   kvlen, nhead, nkvh, dim, vdim, scale);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::self_attention(attn_val->data(), q->data(), k->data(), v->data(), attn_val->dtype(), qlen,
                                      kvlen, nhead, nkvh, dim, vdim, scale);
#endif
#ifdef ENABLE_ILUVATAR_API
    case LLAISYS_DEVICE_ILUVATAR:
        return nvidia::self_attention(attn_val->data(), q->data(), k->data(), v->data(), attn_val->dtype(), qlen,
                                      kvlen, nhead, nkvh, dim, vdim, scale);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

void self_attention_segmented(tensor_t attn_val,
                              tensor_t q,
                              tensor_t k,
                              tensor_t v,
                              float scale,
                              const int64_t *q_offsets,
                              const int64_t *kv_offsets,
                              size_t nseg) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    ASSERT(nseg > 0, "SelfAttentionSegmented: nseg must be > 0.");
    ASSERT(q_offsets && kv_offsets, "SelfAttentionSegmented: offsets must not be null.");

    ASSERT(attn_val->ndim() == 3 && q->ndim() == 3 && k->ndim() == 3 && v->ndim() == 3,
           "SelfAttentionSegmented: all tensors must be 3D.");

    size_t qlen = q->shape()[0];
    size_t nhead = q->shape()[1];
    size_t dim = q->shape()[2];
    size_t kvlen = k->shape()[0];
    size_t nkvh = k->shape()[1];
    size_t kdim = k->shape()[2];
    size_t vdim = v->shape()[2];

    ASSERT(dim == kdim, "SelfAttentionSegmented: q and k head dim mismatch.");
    ASSERT(v->shape()[0] == kvlen && v->shape()[1] == nkvh, "SelfAttentionSegmented: v shape mismatch with k.");
    ASSERT(attn_val->shape()[0] == qlen && attn_val->shape()[1] == nhead && attn_val->shape()[2] == vdim,
           "SelfAttentionSegmented: output shape mismatch.");
    ASSERT(nhead % nkvh == 0, "SelfAttentionSegmented: nhead must be divisible by nkvh.");
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "SelfAttentionSegmented: tensors must be contiguous.");

    ASSERT(q_offsets[0] == 0 && kv_offsets[0] == 0, "SelfAttentionSegmented: offsets must start at 0.");
    ASSERT(static_cast<size_t>(q_offsets[nseg]) == qlen, "SelfAttentionSegmented: q_offsets end mismatch.");
    ASSERT(static_cast<size_t>(kv_offsets[nseg]) == kvlen, "SelfAttentionSegmented: kv_offsets end mismatch.");
    for (size_t i = 0; i < nseg; ++i) {
        ASSERT(q_offsets[i] <= q_offsets[i + 1], "SelfAttentionSegmented: q_offsets must be non-decreasing.");
        ASSERT(kv_offsets[i] <= kv_offsets[i + 1], "SelfAttentionSegmented: kv_offsets must be non-decreasing.");
        const int64_t qseg = q_offsets[i + 1] - q_offsets[i];
        const int64_t kvseg = kv_offsets[i + 1] - kv_offsets[i];
        ASSERT(qseg >= 0 && kvseg >= 0, "SelfAttentionSegmented: invalid negative segment length.");
        ASSERT(kvseg >= qseg, "SelfAttentionSegmented: each segment must satisfy kv_len >= q_len.");
    }

    // Segment-by-segment execution. This preserves correctness on all backends
    // (including NVIDIA) before a fused segmented kernel is introduced.
    for (size_t i = 0; i < nseg; ++i) {
        const size_t qb = static_cast<size_t>(q_offsets[i]);
        const size_t qe = static_cast<size_t>(q_offsets[i + 1]);
        const size_t kb = static_cast<size_t>(kv_offsets[i]);
        const size_t ke = static_cast<size_t>(kv_offsets[i + 1]);
        auto out_seg = attn_val->slice(0, qb, qe);
        auto q_seg = q->slice(0, qb, qe);
        auto k_seg = k->slice(0, kb, ke);
        auto v_seg = v->slice(0, kb, ke);
        self_attention(out_seg, q_seg, k_seg, v_seg, scale);
    }
}
} // namespace llaisys::ops
