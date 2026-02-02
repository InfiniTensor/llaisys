#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_ARGUMENT(attn_val->ndim() == 3, "attn_val must be a 3D tensor");
    CHECK_ARGUMENT(q->ndim() == 3, "q must be a 3D tensor");
    CHECK_ARGUMENT(k->ndim() == 3, "k must be a 3D tensor");
    CHECK_ARGUMENT(v->ndim() == 3, "v must be a 3D tensor");
    CHECK_ARGUMENT(attn_val->shape()[0] == q->shape()[0], "attn_val and q must have the same sequence length");
    CHECK_ARGUMENT(attn_val->shape()[1] == q->shape()[1], "attn_val and q must have the same number of heads");
    CHECK_ARGUMENT(attn_val->shape()[2] == v->shape()[2], "attn_val and v must have the same head dimension");
    CHECK_ARGUMENT(q->shape()[2] == k->shape()[2], "q and k must have the same head dimension");
    CHECK_ARGUMENT(k->shape()[1] == v->shape()[1], "k and v must have the same number of heads");
    CHECK_ARGUMENT(k->shape()[2] == v->shape()[2], "k and v must have the same head dimension");
    CHECK_ARGUMENT(q->shape()[1] % k->shape()[1] == 0, "q's number of heads must be a multiple of k's number of heads");
    CHECK_ARGUMENT(attn_val->isContiguous(), "self_attention: attn_val tensor must be contiguous.");
    CHECK_ARGUMENT(q->isContiguous(), "self_attention: q tensor must be contiguous.");
    CHECK_ARGUMENT(k->isContiguous(), "self_attention: k tensor must be contiguous.");
    CHECK_ARGUMENT(v->isContiguous(), "self_attention: v tensor must be contiguous.");

    size_t seq_len = q->shape()[0];
    size_t num_heads = q->shape()[1];
    size_t num_kv_heads = k->shape()[1];
    size_t head_dim = q->shape()[2];
    size_t total_len = k->shape()[0];

    // 总是支持CPU计算
    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(
            attn_val->data(), 
            q->data(), 
            k->data(), 
            v->data(), 
            scale, 
            attn_val->dtype(), 
            seq_len, 
            num_heads, 
            num_kv_heads, 
            head_dim, 
            total_len
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
            scale, 
            attn_val->dtype(), 
            seq_len, 
            num_heads, 
            num_kv_heads, 
            head_dim, 
            total_len
        );
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
