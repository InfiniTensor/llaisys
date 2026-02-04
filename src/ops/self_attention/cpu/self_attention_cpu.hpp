#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, llaisysDataType_t type, const size_t seqlen, const size_t nhead, const size_t nkvhead, const size_t d, const size_t dv, const size_t token_len, float scale);
}