#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::nvidia {

// Flash Attention 2 CUDA 实现（shared memory tiling，online softmax）
// 支持 GQA (n_heads % n_kv_heads == 0)
// 支持 F32 / F16 / BF16
void flash_attention(std::byte *attn_val,
                     const std::byte *q, const std::byte *k, const std::byte *v,
                     llaisysDataType_t type,
                     size_t seq_len, size_t total_len,
                     size_t n_heads, size_t n_kv_heads,
                     size_t head_dim, float scale);

} // namespace llaisys::ops::nvidia
