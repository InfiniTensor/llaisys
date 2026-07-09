#include "op.hpp"
#include "../../utils.hpp"
#include "./cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
    
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    // 1. 基础校验
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());

    // 2. 形状提取
    size_t seqlen = q->shape()[0];
    size_t nhead = q->shape()[1];
    size_t d = q->shape()[2];

    size_t total_len = k->shape()[0];
    size_t nkvhead = k->shape()[1];
    // k 的 d 必须与 q 的 d 一致
    
    size_t dv = v->shape()[2];

    // 3. 约束校验
    CHECK_ARGUMENT(nhead % nkvhead == 0, "nhead must be divisible by nkvhead (GQA)");
    CHECK_ARGUMENT(k->shape()[2] == d, "Q and K head_dim mismatch");
    CHECK_ARGUMENT(v->shape()[0] == total_len, "K and V total_len mismatch");
    CHECK_ARGUMENT(v->shape()[1] == nkvhead, "K and V nkvhead mismatch");

    // 4. 设备分发
    if (q->deviceType() == LLAISYS_DEVICE_CPU) {
        cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                            q->dtype(), seqlen, total_len, nhead, nkvhead, d, dv, scale);
        return;
    }

    TO_BE_IMPLEMENTED();
}
} // namespace llaisys::ops
