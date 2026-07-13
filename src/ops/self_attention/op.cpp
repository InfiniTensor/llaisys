#include "op.hpp"
#include "cpu/self_attention_cpu.hpp"
#include "../../core/context/context.hpp"
#include "../../utils.hpp"
#include <stdexcept>

namespace llaisys::ops::nvidia {
#ifdef ENABLE_NVIDIA_API
    void self_attention(tensor_t output, tensor_t q, tensor_t k, tensor_t v, float scale);
#endif
}

namespace llaisys::ops {

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q);
    CHECK_SAME_DEVICE(attn_val, k);
    CHECK_SAME_DEVICE(attn_val, v);

    auto device = core::context().runtime().deviceType();

    size_t seqlen = q->shape()[0];
    size_t nhead  = q->shape()[1];
    size_t d      = q->shape()[2];

    size_t total_len = k->shape()[0];
    size_t nkvhead   = k->shape()[1];
    size_t dv        = v->shape()[2];

    if (nhead % nkvhead != 0) {
        throw std::invalid_argument("SelfAttention: nhead must be divisible by nkvhead");
    }
    if (k->shape()[2] != d) {
        throw std::invalid_argument("SelfAttention: Key dimension must match Query dimension");
    }
    if (attn_val->shape()[0] != seqlen || 
        attn_val->shape()[1] != nhead || 
        attn_val->shape()[2] != dv) {
         throw std::invalid_argument("SelfAttention: Output shape mismatch");
    }

    if (device == LLAISYS_DEVICE_CPU) {
        cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                            q->dtype(), seqlen, total_len, nhead, nkvhead, d, dv, scale);
    } 
#ifdef ENABLE_NVIDIA_API
    else if (device == LLAISYS_DEVICE_NVIDIA) {
        nvidia::self_attention(attn_val, q, k, v, scale);
    }
#endif
    else {
        throw std::runtime_error("SelfAttention: Unsupported device physical routing");
    }
}

} // namespace llaisys::ops