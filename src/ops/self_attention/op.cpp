#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/self_attension_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    // Only support contiguous inputs for now.
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(), "Self-Attention: all tensors must be contiguous.");
    // attn_val, q, k, v should have same ndim and should be 3D tensors
    CHECK_SAME(attn_val->ndim(), q->ndim(), k->ndim(), v->ndim());
    ASSERT(attn_val->ndim() == 3, "Self-Attention: only support 3D tensors for now.");
    // attn_val shape dim 0,1 is same as q shape
    CHECK_SAME_SHAPE(attn_val->shape()[0], q->shape()[0]);
    CHECK_SAME_SHAPE(attn_val->shape()[1], q->shape()[1]);

    // k, v shape dim 0 is kvlen; allow kvlen != qlen
    CHECK_SAME_SHAPE(k->shape()[0], v->shape()[0]);

    // q and k must share head dim; output matches v head dim
    CHECK_SAME_SHAPE(k->shape()[2], q->shape()[2]);
    CHECK_SAME_SHAPE(attn_val->shape()[2], v->shape()[2]);

	// nhead and kv_head should divide evenly
	ASSERT(q->shape()[1] % k->shape()[1] == 0, "Self-Attention: nhead must be divisible by kv_head.");

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), attn_val->dtype(), q->shape()[0], k->shape()[0], q->shape()[1], k->shape()[1], q->shape()[2], v->shape()[2], scale);
        return;
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), attn_val->dtype(), q->shape()[0], k->shape()[0], q->shape()[1], k->shape()[1], q->shape()[2], v->shape()[2], scale);
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
