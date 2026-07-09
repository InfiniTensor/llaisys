#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::cpu {

// Flash Attention 2 (CPU, tiled online-softmax)
// Supports Grouped Query Attention (GQA): n_heads % n_kv_heads == 0
// Layout: Q[seq_len, n_heads, head_dim]
//         K[total_len, n_kv_heads, head_dim]
//         V[total_len, n_kv_heads, head_dim]
//       out[seq_len, n_heads, head_dim]
void flash_attention(std::byte *out, const std::byte *q, const std::byte *k, const std::byte *v,
                     llaisysDataType_t type,
                     size_t seq_len, size_t total_len,
                     size_t n_heads, size_t n_kv_heads, size_t head_dim, float scale);

} // namespace llaisys::ops::cpu
