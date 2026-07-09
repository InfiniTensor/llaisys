#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    // q: (seq_len, n_heads, head_dim)
    // k: (total_len, n_kv_heads, head_dim)
    // v: (total_len, n_kv_heads, v_dim)
    // attn_val: (seq_len, n_heads, v_dim)

    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());

    ASSERT(q->ndim() == 3, "q must be 3D");
    ASSERT(k->ndim() == 3, "k must be 3D");
    ASSERT(v->ndim() == 3, "v must be 3D");
    ASSERT(attn_val->ndim() == 3, "attn_val must be 3D");

    size_t seq_len = q->shape()[0];
    size_t n_heads = q->shape()[1];
    size_t head_dim = q->shape()[2];

    size_t total_len = k->shape()[0];
    size_t n_kv_heads = k->shape()[1];
    ASSERT(k->shape()[2] == head_dim, "k head_dim mismatch");

    ASSERT(v->shape()[0] == total_len, "v total_len mismatch");
    ASSERT(v->shape()[1] == n_kv_heads, "v n_kv_heads mismatch");
    size_t v_dim = v->shape()[2];

    ASSERT(attn_val->shape()[0] == seq_len, "attn_val seq_len mismatch");
    ASSERT(attn_val->shape()[1] == n_heads, "attn_val n_heads mismatch");
    ASSERT(attn_val->shape()[2] == v_dim, "attn_val v_dim mismatch");

    ASSERT(n_heads % n_kv_heads == 0, "n_heads must be multiple of n_kv_heads");

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                   attn_val->dtype(), seq_len, total_len,
                                   n_heads, n_kv_heads, head_dim, v_dim, scale);
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
