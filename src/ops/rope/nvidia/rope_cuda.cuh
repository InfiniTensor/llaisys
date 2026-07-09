#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
// RoPE CUDA 实现
// 每个 thread 处理一个 (sequence_pos, head, dim_pair) 三元组
// 充分利用 A100 的线程级并行度
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          llaisysDataType_t type, size_t seq_len, size_t n_heads,
          size_t head_dim, float theta);
} // namespace llaisys::ops::nvidia
