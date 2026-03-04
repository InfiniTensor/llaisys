#pragma once

#include "llaisys.h"

namespace llaisys::ops::nvidia {

void self_attention(
    std::byte *out, 
    const std::byte *q, 
    const std::byte *k, 
    const std::byte *v,
    llaisysDataType_t dtype,
    size_t q_len, 
    size_t kv_len,
    size_t num_q_heads, 
    size_t num_kv_heads,
    size_t head_dim_qk, 
    size_t head_dim_v,
    float softmax_scale
);

} // namespace llaisys::ops::nvidia