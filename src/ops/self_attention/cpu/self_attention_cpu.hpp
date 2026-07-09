#pragma once
#include "../../../core/llaisys_core.hpp"

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t dtype, size_t seq_len, size_t total_len,
                    size_t n_heads, size_t n_kv_heads, size_t head_dim, size_t v_dim, float scale);
}
