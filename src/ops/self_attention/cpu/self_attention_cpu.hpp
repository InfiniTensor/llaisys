#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
// Implements causal self-attention:
//   out = softmax((q @ k^T) * scale + causal_mask) @ v
//
// Layout expectations (contiguous):
//   q   : [qlen, nh,   hd]
//   k/v : [kvlen, nkvh, hd]   (GQA/MQA supported when nh % nkvh == 0)
//   out : [qlen, nh,   hd]
//
// causal_mask matches test/ops/self_attention.py:
//   allow key index j <= i + (kvlen - qlen)
void self_attention(std::byte *out,
                    const std::byte *q,
                    const std::byte *k,
                    const std::byte *v,
                    llaisysDataType_t dtype,
                    size_t qlen,
                    size_t nh,
                    size_t kvlen,
                    size_t nkvh,
                    size_t hd,
                    float scale);
} // namespace llaisys::ops::cpu
