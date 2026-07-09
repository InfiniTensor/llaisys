#include "op.hpp"

#include "../../utils.hpp"

#include "cpu/self_attention.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    
    size_t seqlen = q->shape()[0];
    size_t nhead = q->shape()[1];
    size_t d = q->shape()[2];
    size_t total_len = k->shape()[0];
    size_t nkvhead = k->shape()[1];
    size_t dv = v->shape()[2];

    if(attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(
            attn_val->data(), q->data(), k->data(), v->data(),
            seqlen, nhead, d,
            total_len, nkvhead, dv,
            scale,
            q->dtype()
        );
    }
    TO_BE_IMPLEMENTED();
}
} // namespace llaisys::ops
