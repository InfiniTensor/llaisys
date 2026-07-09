#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::cpu {

void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    size_t seqlen, size_t total_len, 
                    size_t nhead, size_t nkvhead, 
                    size_t d, size_t dv,
                    float scale,
                    llaisysDataType_t dtype);

} // namespace llaisys::ops::cpu