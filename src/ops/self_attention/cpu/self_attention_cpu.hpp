#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                   llaisysDataType_t type, size_t seqlen, size_t totlen, size_t nh, size_t nkvh, 
                   size_t d, size_t dv, float scale);
}
