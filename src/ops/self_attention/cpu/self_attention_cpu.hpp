#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    float scale, llaisysDataType_t dtype,
                    size_t seq_len, size_t total_len, size_t nhead, size_t nkvhead, size_t d, size_t dv);
}
