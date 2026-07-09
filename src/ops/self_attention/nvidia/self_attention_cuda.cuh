#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
// Self-Attention CUDA 实现（支持 GQA）
// 流程：
//   1. 对每个 head，用 cuBLAS 计算 S = Q * K^T * scale
//   2. 用自定义 kernel 对 S 做因果 softmax
//   3. 用 cuBLAS 计算 O = softmax(S) * V
void self_attention(std::byte *attn_val,
                    const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t type,
                    size_t seq_len, size_t total_len,
                    size_t n_heads, size_t n_kv_heads,
                    size_t head_dim, float scale);
} // namespace llaisys::ops::nvidia
